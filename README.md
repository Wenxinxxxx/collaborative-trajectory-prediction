# Collaborative Vehicle Trajectory Prediction Based on Deep Learning

基于深度学习的协同车辆轨迹预测 — 毕业设计项目

---

## 项目概述

本项目实现了 **7个** 轨迹预测模型的对比分析，包括4个非协同模型和3个协同模型（含1个自主改进模型），在V2X-Seq TFD数据集上进行实验评估。

---

## 模型列表

| 编号 | 模型名称 | 命令行名称 | 类型 | 核心技术 |
|------|---------|-----------|------|----------|
| 1 | LSTM Encoder-Decoder | `lstm_seq2seq` | 非协同基线 | LSTM编码器-解码器，多模态输出 |
| 2 | Social LSTM | `social_lstm` | 非协同 | 社交池化建模邻居交互 |
| 3 | GRIP++ | `grip_plus` | 非协同 | 图卷积交互感知预测 |
| 4 | Transformer | `transformer` | 非协同 | 自注意力 + 交叉注意力 |
| 5 | V2X-Graph | `v2x_graph` | 协同 | 图注意力网络 + V2X融合 |
| 6 | Co-MTP | `co_mtp` | 协同 | 多时间尺度 + GNN + Transformer |
| 7 | Enhanced Co-MTP | `enhanced_co_mtp` | 协同(改进) | 跨代理注意力融合 + 多尺度时间编码 + 可变形注意力 |

---

## 项目结构

```
collaborative_trajectory_prediction/
├── config.py                    # 全局配置（超参数、路径等）
├── setup.sh                     # 一键环境部署脚本
├── requirements.txt             # Python依赖
├── quick_test.py                # 快速验证脚本（2 epochs）
├── run_experiments.py           # 一键运行完整实验
├── train.py                     # 统一训练脚本
├── evaluate.py                  # 评估脚本
├── visualize.py                 # 可视化脚本
├── data/
│   ├── preprocess.py            # 数据预处理
│   ├── generate_synthetic.py    # 模拟数据生成（调试用）
│   └── download_v2x_seq.py      # 数据集下载指引
├── datasets/
│   ├── trajectory_dataset.py    # PyTorch Dataset类
│   └── processed/               # 预处理后的数据
├── models/
│   ├── __init__.py              # 模型注册
│   ├── lstm_seq2seq.py          # LSTM Encoder-Decoder
│   ├── social_lstm.py           # Social LSTM
│   ├── grip_plus.py             # GRIP++
│   ├── transformer_pred.py      # Transformer Predictor
│   ├── v2x_graph.py             # V2X-Graph
│   ├── co_mtp.py                # Co-MTP
│   └── enhanced_co_mtp.py       # Enhanced Co-MTP（改进模型）
├── utils/
│   ├── metrics.py               # 评估指标（ADE/FDE/MR）
│   ├── losses.py                # 损失函数
│   └── helpers.py               # 辅助工具
├── checkpoints/                 # 模型权重
├── results/                     # 评估结果（JSON）
└── visualizations/              # 可视化图表（PNG）
```

---

## 快速开始

### 1. 环境部署

```bash
cd /data/yb/collaborative_trajectory_prediction
bash setup.sh
conda activate traj_pred
```

### 2. 数据准备

**数据集来源：** 本项目使用的 V2X-Seq-TFD 数据集由清华大学AIR实验室提供，原始数据下载地址：

> [V2X-Seq-TFD Dataset (Google Drive)](https://drive.google.com/drive/folders/1yDnlrPCKImpVfI1OPBYyzLFWkhZP5v-7)

```bash
# 将 V2X-Seq-TFD-Example.zip 放到 data/ 目录下
cd data
unzip V2X-Seq-TFD-Example.zip
cd ..

# 预处理
python data/preprocess.py --data v2x_seq --mode cooperative
```

### 3. 快速验证

```bash
python quick_test.py
```

### 4. 完整训练

```bash
# 训练所有7个模型
python run_experiments.py

# 训练单个模型
python train.py --model enhanced_co_mtp

# 快速训练（5 epochs，调试用）
python run_experiments.py --quick
```

### 5. 评估和可视化

```bash
python evaluate.py
python visualize.py
```

---

## 评估指标

| 指标 | 全称 | 含义 |
|------|------|------|
| **minADE** | Minimum Average Displacement Error | 最佳预测轨迹的平均位移误差（米） |
| **minFDE** | Minimum Final Displacement Error | 最佳预测轨迹的终点位移误差（米） |
| **MR** | Miss Rate | 终点误差超过阈值的比例 |

所有指标越低越好。

---

## 数据集声明

本项目使用的 **V2X-Seq-TFD** 数据集来自清华大学AIR实验室（THU AIR Lab），该数据集为车路协同感知与预测领域的大规模序列数据集。

- **数据集下载：** [Google Drive](https://drive.google.com/drive/folders/1yDnlrPCKImpVfI1OPBYyzLFWkhZP5v-7)
- **原始论文：** Yu et al., "V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting," CVPR 2023.
- **数据集许可：** 该数据集仅用于学术研究目的，使用时请遵循原作者的许可协议并引用原始论文。

---

## 参考文献

1. Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
2. Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces," CVPR 2016.
3. Li et al., "GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction," arXiv 2019.
4. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
5. Yu et al., "V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting," CVPR 2023.
