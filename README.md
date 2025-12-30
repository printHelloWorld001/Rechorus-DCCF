# DCCF 模型在 ReChorus 框架下的复现与性能边界分析

本项目基于 **ReChorus** 统一推荐框架复现了去中心化意图解耦对比学习模型 (**DCCF**) 。旨在通过意图解耦机制有效提升模型在复杂推荐场景中的鲁棒性。

## 🌟 核心功能
* **意图解耦机制**: 通过引入 $K$ 个可学习的全局意图原型捕获用户多维偏好，有效缓解传统图卷积模型中的过度平滑问题。
* **自适应掩码生成**: 基于节点嵌入计算用户与物品的余弦相似度 $sim(u,i)$，动态生成增强图结构 $G'$ 以抑制交互噪声。
* **工程化显存优化**: 本项目实现了基于 **分块 (Chunk-based)** 的流式推理算法，通过显存预判与显式垃圾回收 (GC) 协同工作，在不损失计算精度的前提下将显存峰值降低了约 40% 。

## 📊 实验结果
我们选取了稠密数据集 (ML-1M) 与稀疏数据集 (Amazon-Grocery) 进行对比实验，评估模型在不同场景下的性能表现：

### 1. 模型性能总表

> **注**：表中<b><code>加粗且带背景</code></b> 的数值为该数据集下的最优结果。
<table>
  <thead>
    <tr>
      <th align="left">Model</th>
      <th align="center">HR@5</th>
      <th align="center">NDCG@5</th>
      <th align="center">HR@10</th>
      <th align="center">NDCG@10</th>
      <th align="center">HR@20</th>
      <th align="center">NDCG@20</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #f6f8fa;">
      <td colspan="7" align="center"><b>Dataset:  🎞️ ML-1M </b></td>
    </tr>
    <tr>
      <td align="left">BPRMF</td>
      <td align="center"><b><code>0.3859</code></b></td>
      <td align="center"><b><code>0.2625</code></b></td>
      <td align="center"><b><code>0.5536</code></b></td>
      <td align="center"><b><code>0.3162</code></b></td>
      <td align="center"><b><code>0.7526</code></b></td>
      <td align="center"><b><code>0.3665</code></b></td>
    </tr>
    <tr>
      <td align="left">LightGCN</td>
      <td align="center">0.3660</td>
      <td align="center">0.2450</td>
      <td align="center">0.5261</td>
      <td align="center">0.2966</td>
      <td align="center">0.7276</td>
      <td align="center">0.3474</td>
    </tr>
    <tr>
      <td align="left">DCCF</td>
      <td align="center">0.2866</td>
      <td align="center">0.1974</td>
      <td align="center">0.4288</td>
      <td align="center">0.2741</td>
      <td align="center">0.6917</td>
      <td align="center">0.2973</td>
    </tr>
    <tr style="background-color: #f6f8fa;">
      <td colspan="7" align="center"><b>Dataset: 🛒 Grocery_and_Gourmet_Food</b></td>
    </tr>
    <tr>
      <td align="left">BPRMF</td>
      <td align="center">0.3238</td>
      <td align="center">0.2233</td>
      <td align="center">0.4342</td>
      <td align="center">0.2592</td>
      <td align="center">0.5479</td>
      <td align="center">0.2877</td>
    </tr>
    <tr>
      <td align="left">LightGCN</td>
      <td align="center"><b><code>0.3708</code></b></td>
      <td align="center"><b><code>0.2542</code></b></td>
      <td align="center"><b><code>0.4974</code></b></td>
      <td align="center"><b><code>0.2954</code></b></td>
      <td align="center"><b><code>0.6156</code></b></td>
      <td align="center"><b><code>0.3252</code></b></td>
    </tr>
    <tr>
      <td align="left">DCCF (Best)</td>
      <td align="center">0.3504</td>
      <td align="center">0.2444</td>
      <td align="center">0.4633</td>
      <td align="center">0.2811</td>
      <td align="center">0.5677</td>
      <td align="center">0.3075</td>
    </tr>
  </tbody>
</table>

### 2. 结论摘要
| 数据集 | 特性 | 结论 |
| :--- | :--- | :--- |
| **ML-1M (稠密)** |交互高度稠密 (4.47%)|**性能退化**：强协同信号下意图过度解耦稀释了全局信号，且掩码机制容易产生“错误剪枝”并误删真实偏好。 |
| **Amazon-Grocery (稀疏)** |极其稀疏，长尾商品多 |**优势展现**：图卷积结构与自适应增强机制有效缓解了数据稀疏性，模型表现优于基础 BPRMF。 |

## 🛠️ 安装与运行

### 1. 环境准备 (Environment Setup)
本项目基于 **Python 3.10** 开发。建议使用 Conda 创建独立环境以确保依赖兼容性。

**第一步：克隆仓库**
```bash
git clone https://github.com/printHelloWorld001/Rechorus-DCCF.git
cd Rechorus-DCCF
```
**第二步：创建并激活环境**
```bash
# 1. 创建虚拟环境
conda create -n dccf_env python=3.10
conda activate dccf_env

# 2. 安装依赖
# 注意：如果安装 torch-scatter 失败，请参考 PyG 官网根据 CUDA 版本手动安装
pip install -r requirements.txt
```
### 2. 模型训练 (Training)
本复现代码已集成至 src/models/general/ 目录。我们提供了针对 稠密数据 (ML-1M) 和 稀疏数据 (Amazon Grocery) 两套经过验证的超参数配置。
#### 📌 场景一：ML-1M Top-K (稠密数据)

使用以下指令进行训练：

```bash
python src/main.py --model_name DCCF --dataset ML_1MTOPK --num_workers 0 --epoch 20 --lr 1e-3 --emb_size 64 --n_layers 2 --n_intents 4
```

#### 📌 场景二：Amazon Grocery (稀疏数据)

使用以下指令进行训练：

```bash
python src/main.py --model_name DCCF --dataset Grocery_and_Gourmet_Food --test_all 0 --emb_size 64 --epoch 20 --lr 1e-4 --l2 1e-4 --ssl_reg 0.0001 --cen_reg 0.001 --n_intents 4 --num_workers 0
```

## 🧪 消融实验 (Ablation Study)
我们在 Amazon-Grocery 数据集上考察了核心组件对推荐质量的具体贡献：

| 模型变体 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| :--- | :---: | :---: | :---: | :---: |
| 完整模型 (DCCF) | 0.4648 | **0.2810** | 0.5763 | **0.3091** |
| -DME (移除意图解耦) | **0.4759** | 0.2801 | **0.5878** | 0.3082 |
| -LocalR (移除本地掩码) | 0.4639 | 0.2796 | 0.5703 | 0.3064 |
| -DisenR (移除意图掩码) | 0.4635 | 0.2794 | 0.5714 | 0.3066 |
| -AllAda (移除增强对比) | 0.4654 | 0.2812 | 0.5502 | 0.2917 |

### 消融实验复现命令
**基线指标 (Full Model)**:
```bash
python src/main.py --model_name DCCF_Ablation --dataset Grocery_and_Gourmet_Food --emb_size 64 --epoch 20 --lr 1e-4 --l2 1e-4 --ssl_reg 1e-5 --cen_reg 1e-4 --n_intents 4 --num_workers 0 --ablation none
```

* (i) 解耦多意图编码消融（-Disen / DME）:考察意图原型对挖掘细粒度潜在偏好的作用。
```bash
python src/main.py --model_name DCCF_Ablation --dataset Grocery_and_Gourmet_Food --emb_size 64 --epoch 20 --lr 1e-4 --l2 1e-4 --ssl_reg 1e-5 --cen_reg 1e-4 --n_intents 4 --num_workers 0 --ablation DME
```

(ii) 参数化自适应掩码 (PAM)；
意图自适应掩码消融（-DisenR / PAM-Disen）:考察自适应剪枝对冗余噪声的识别作用 。
```bash
python src/main.py --model_name DCCF_Ablation --dataset Grocery_and_Gourmet_Food --emb_size 64 --epoch 20 --lr 1e-4 --l2 1e-4 --ssl_reg 1e-5 --cen_reg 1e-4 --n_intents 4 --num_workers 0 --ablation DisenR
```

(iii) 自监督学习 (SSL)：察自监督信号对模型稳定性的支撑作用 。

全局自监督信号消融（-DisenG / SSL-Global）:
```bash
python src/main.py --model_name DCCF_Ablation --dataset Grocery_and_Gourmet_Food --emb_size 64 --epoch 20 --lr 1e-4 --l2 1e-4 --ssl_reg 1e-5 --cen_reg 1e-4 --n_intents 4 --num_workers 0 --ablation DisenG
```

增强自监督信号消融（-AllAda / SSL-Augment）:
```bash
python src/main.py --model_name DCCF_Ablation --dataset Grocery_and_Gourmet_Food --emb_size 64 --epoch 20 --lr 1e-4 --l2 1e-4 --ssl_reg 1e-5 --cen_reg 1e-4 --n_intents 4 --num_workers 0 --ablation AllAda
```


