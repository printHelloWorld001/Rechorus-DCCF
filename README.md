# DCCF 模型在 ReChorus 框架下的复现与优化

[cite_start]本项目基于 **ReChorus** 统一推荐框架复现了去中心化意图解耦对比学习模型 (**DCCF**) [cite: 7, 19]，并针对大规模图计算进行了工程优化。

## 🌟 核心功能
* [cite_start]**意图解耦机制**: 通过全局意图原型捕获用户多维偏好，缓解过度平滑问题 [cite: 18, 23]。
* [cite_start]**自适应掩码生成**: 基于节点嵌入相似度动态剪枝，抑制交互噪声 [cite: 30]。
* [cite_start]**工程优化**: 实现了基于 **分块 (Chunk-based)** 的流式推理算法，有效降低显存峰值占用约 40% [cite: 8, 62]。

## 📊 实验结果
[cite_start]我们在两个不同特性的数据集上评估了模型表现 [cite: 83, 96]：
| 数据集 | 结论 |
| :--- | :--- |
| **ML-1M (稠密)** | [cite_start]存在性能退化，主因是意图过度解耦稀释了强协同信号。 |
| **Amazon-Grocery (稀疏)** | [cite_start]表现出较强优势，自适应增强机制有效缓解了数据稀疏性 [cite: 8, 124]。 |

## 🛠️ 安装与运行
1. 克隆本项目并安装 ReChorus 环境。
2. [cite_start]将本模型代码放入 `src/models/general/` 目录。
3. 运行示例命令：
```bash
python main.py --model DCCF --dataset Amazon-Grocery --n_intents 4 --ssl_reg 1e-5
