# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import gc

try:
    import torch_sparse
except ImportError:
    torch_sparse = None
    print("[Warning] 'torch_sparse' not found. DCCF requires torch-sparse library.")

from models.BaseModel import GeneralModel

class DCCF3(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'n_layers', 'n_intents', 'ssl_temp', 'ssl_reg', 'cen_reg']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64, help='Size of embedding vectors.')
        parser.add_argument('--n_layers', type=int, default=2, help='Number of graph convolution layers.')
        parser.add_argument('--n_intents', type=int, default=16, help='Number of latent intents (K).')
        parser.add_argument('--ssl_temp', type=float, default=0.1, help='Temperature for contrastive loss.')
        parser.add_argument('--ssl_reg', type=float, default=0.03, help='Weight for contrastive loss.')
        parser.add_argument('--cen_reg', type=float, default=0.001, help='Weight for intent centering loss.')
        parser.add_argument('--emb_reg', type=float, default=1e-4, help='Weight for L2 regularization.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.emb_size
        self.n_layers = args.n_layers
        self.n_intents = args.n_intents
        self.temp = args.ssl_temp
        self.ssl_reg = args.ssl_reg
        self.cen_reg = args.cen_reg
        self.emb_reg = args.emb_reg

        self._prepare_graph_structure(corpus)

        self.user_embedding = nn.Embedding(self.user_num, self.emb_size)
        self.item_embedding = nn.Embedding(self.item_num, self.emb_size)

        _user_intent = torch.empty(self.emb_size, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)

        _item_intent = torch.empty(self.emb_size, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

        self.apply(self.init_weights)
        self.ua_embedding = None
        self.ia_embedding = None

    @staticmethod
    def build_adjmat(user_count, item_count, train_mat):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1
        R = R.tolil()

        adj_mat = sp.dok_matrix((user_count + item_count, user_count + item_count), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:user_count, user_count:] = R
        adj_mat[user_count:, :user_count] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(1)) + 1e-10
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()

    # def _prepare_graph_structure(self, corpus):
    #     norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
    #     self.all_h_list = torch.LongTensor(norm_adj.row)
    #     self.all_t_list = torch.LongTensor(norm_adj.col)
    #     self.G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)
    #     self.G_values = torch.FloatTensor(norm_adj.data)
    #     self.A_in_shape = norm_adj.shape
    def _prepare_graph_structure(self, corpus):
        norm_adj = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        
        # 将数据转换为 Tensor
        indices = torch.stack([torch.LongTensor(norm_adj.row), torch.LongTensor(norm_adj.col)], dim=0)
        values = torch.FloatTensor(norm_adj.data)
        h_list = torch.LongTensor(norm_adj.row)
        t_list = torch.LongTensor(norm_adj.col)

        # 使用 register_buffer，这样它们会随模型一起 .to(cuda) 或 .to(cpu)
        self.register_buffer('G_indices', indices)
        self.register_buffer('G_values', values)
        self.register_buffer('all_h_list', h_list)
        self.register_buffer('all_t_list', t_list)
        
        self.A_in_shape = norm_adj.shape

    # def _adaptive_mask(self, all_embeddings, h_list, t_list):
    #     # 增大 chunk_size 利用 CPU 内存
    #     chunk_size = 300000 
    #     total_edges = h_list.shape[0]
    #     values_list = []
        
    #     with torch.no_grad():
    #         for start in range(0, total_edges, chunk_size):
    #             end = min(start + chunk_size, total_edges)
    #             h_idx = h_list[start:end]
    #             t_idx = t_list[start:end]
                
    #             # CPU 计算
    #             head_emb = all_embeddings[h_idx].cpu()
    #             tail_emb = all_embeddings[t_idx].cpu()
                
    #             head_norm = F.normalize(head_emb, p=2, dim=1)
    #             tail_norm = F.normalize(tail_emb, p=2, dim=1)
                
    #             # 相似度计算
    #             edge_val = (torch.sum(head_norm * tail_norm, dim=1) + 1) / 2
    #             values_list.append(edge_val)
                
    #     edge_alpha = torch.cat(values_list, dim=0)
        
    #     A_tensor = torch_sparse.SparseTensor(
    #         row=self.all_h_list, col=self.all_t_list, value=edge_alpha, 
    #         sparse_sizes=self.A_in_shape
    #     )
    #     D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
    #     G_values = D_scores_inv[self.all_h_list] * edge_alpha
        
    #     return self.G_indices, G_values
    def _adaptive_mask(self, all_embeddings, h_list, t_list):
        chunk_size = 300000 
        total_edges = h_list.shape[0]
        values_list = []

        # 核心：自动获取当前模型所在的设备（可能是 cpu 或 cuda）
        device = all_embeddings.device
        
        with torch.no_grad():
            for start in range(0, total_edges, chunk_size):
                end = min(start + chunk_size, total_edges)
                
                # [关键] 确保索引也在同一设备上
                h_idx = h_list[start:end].to(device)
                t_idx = t_list[start:end].to(device)
                
                head_emb = all_embeddings[h_idx]
                tail_emb = all_embeddings[t_idx]
                
                head_norm = F.normalize(head_emb, p=2, dim=1)
                tail_norm = F.normalize(tail_emb, p=2, dim=1) 
                
                edge_val = (torch.sum(head_norm * tail_norm, dim=1) + 1) / 2
                values_list.append(edge_val)
                
        edge_alpha = torch.cat(values_list, dim=0)
        
        # [关键] 构造 SparseTensor 时，确保所有的输入都在同一个 device
        # self.all_h_list 和 self.all_t_list 也要 .to(device)
        A_tensor = torch_sparse.SparseTensor(
            row=self.all_h_list.to(device), 
            col=self.all_t_list.to(device), 
            value=edge_alpha, 
            sparse_sizes=self.A_in_shape
        )
        
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)
        G_values = D_scores_inv[self.all_h_list.to(device)] * edge_alpha
        
        # 返回时，确保 G_indices 也在对应设备上
        return self.G_indices.to(device), G_values
    def _graph_propagation(self):
        all_embeddings = [torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = [], [], [], []

        for i in range(self.n_layers):
            # 1. Local View
            gnn_layer_embeddings = torch_sparse.spmm(
                self.G_indices, self.G_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )

            # 2. Intent View
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.user_num, self.item_num], 0)
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            int_layer_embeddings = torch.cat([u_int_embeddings, i_int_embeddings], dim=0)

            # 3. Adaptive Masking (实时)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_layer_embeddings, self.all_h_list, self.all_t_list)
            gc.collect() 
            
            G_inten_indices, G_inten_values = self._adaptive_mask(int_layer_embeddings, self.all_h_list, self.all_t_list)
            gc.collect()

            # 4. Augmented Views
            gaa_layer_embeddings = torch_sparse.spmm(
                G_graph_indices, G_graph_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )
            iaa_layer_embeddings = torch_sparse.spmm(
                G_inten_indices, G_inten_values, 
                self.A_in_shape[0], self.A_in_shape[1], 
                all_embeddings[i]
            )

            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            gaa_embeddings.append(gaa_layer_embeddings)
            iaa_embeddings.append(iaa_layer_embeddings)

            # Aggregation
            next_layer = gnn_layer_embeddings + int_layer_embeddings + gaa_layer_embeddings + iaa_layer_embeddings + all_embeddings[i]
            all_embeddings.append(next_layer)
        
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)
        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.user_num, self.item_num], 0)
        return gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings

    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)
        cl_loss = 0.0
        
        def cal_loss_fn(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), dim=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            return loss / (pos_score.shape[0] + 1e-8)

        for i in range(len(gnn_emb)):
            u_gnn, i_gnn = torch.split(gnn_emb[i], [self.user_num, self.item_num], 0)
            u_int, i_int = torch.split(int_emb[i], [self.user_num, self.item_num], 0)
            u_gaa, i_gaa = torch.split(gaa_emb[i], [self.user_num, self.item_num], 0)
            u_iaa, i_iaa = torch.split(iaa_emb[i], [self.user_num, self.item_num], 0)

            u_gnn = F.normalize(u_gnn[users], dim=1)
            u_int = F.normalize(u_int[users], dim=1)
            u_gaa = F.normalize(u_gaa[users], dim=1)
            u_iaa = F.normalize(u_iaa[users], dim=1)
            
            i_gnn = F.normalize(i_gnn[items], dim=1)
            i_int = F.normalize(i_int[items], dim=1)
            i_gaa = F.normalize(i_gaa[items], dim=1)
            i_iaa = F.normalize(i_iaa[items], dim=1)

            cl_loss += cal_loss_fn(u_gnn, u_int)
            cl_loss += cal_loss_fn(u_gnn, u_gaa)
            cl_loss += cal_loss_fn(u_gnn, u_iaa)
            cl_loss += cal_loss_fn(i_gnn, i_int)
            cl_loss += cal_loss_fn(i_gnn, i_gaa)
            cl_loss += cal_loss_fn(i_gnn, i_iaa)

        return cl_loss

    def forward(self, feed_dict):
        return self.inference(feed_dict)

    def inference(self, feed_dict):
        user = feed_dict['user_id'].long()
        items = feed_dict['item_id'].long()
        if self.ua_embedding is None:
            self._graph_propagation()
        u_embed = self.ua_embedding[user]
        i_embed = self.ia_embedding[items]
        if items.dim() == 2:
            u_embed = u_embed.unsqueeze(1)
            
        prediction = (u_embed * i_embed).sum(dim=-1)
        
        # [修复] 复制原始字典，防止丢失 user_id 等信息导致 loss 报错
        out_dict = feed_dict.copy()
        out_dict['prediction'] = prediction
        return out_dict

    def loss(self, feed_dict):
        users = feed_dict['user_id'].long()
        if users.dim() > 1: users = users.flatten()
        pos_items = feed_dict['item_id'].long()
        if pos_items.dim() > 1: pos_items = pos_items[:, 0]
        
        if 'neg_items' in feed_dict:
            neg_items = feed_dict['neg_items'].long()
            if neg_items.dim() > 1: neg_items = neg_items[:, 0]
        else:
            neg_items = torch.randint(0, self.item_num, (users.shape[0],), device=users.device)

        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = self._graph_propagation()

        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        neg_embeddings = self.ia_embedding[neg_items]
        
        pos_scores = torch.sum(u_embeddings * pos_embeddings, dim=1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, dim=1)
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + 
                    pos_embeddings_pre.norm(2).pow(2) + 
                    neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss / users.shape[0]

        cen_loss = 0.0
        if self.cen_reg > 0:
            cen_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
            cen_loss = self.cen_reg * cen_loss

        cl_loss = self.ssl_reg * self.cal_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings)

        return mf_loss + emb_loss + cen_loss + cl_loss
