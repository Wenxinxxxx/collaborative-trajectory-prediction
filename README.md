# Collaborative Vehicle Trajectory Prediction Based on Deep Learning

A graduation project comparing single-view and cooperative perception models for vehicle trajectory prediction.

---

## Project Overview

This project implements a comparative analysis of **7 trajectory prediction models**, including 4 single-view models and 3 cooperative perception models (with 1 self-improved model). All experiments are evaluated on the V2X-Seq TFD dataset.

---

## Model List

| No. | Model Name | CLI Name | Type | Core Technique |
|-----|-----------|----------|------|----------------|
| 1 | LSTM Encoder-Decoder | `lstm_seq2seq` | Single-View Baseline | LSTM encoder-decoder with multi-modal output |
| 2 | Social LSTM | `social_lstm` | Single-View | Social pooling for neighbor interaction modeling |
| 3 | GRIP++ | `grip_plus` | Single-View | Graph convolution for interaction-aware prediction |
| 4 | Transformer | `transformer` | Single-View | Self-attention + cross-attention mechanism |
| 5 | V2X-Graph | `v2x_graph` | Cooperative | Graph attention network + V2X fusion |
| 6 | Co-MTP | `co_mtp` | Cooperative | Multi-temporal scale + GNN + Transformer |
| 7 | Enhanced Co-MTP | `enhanced_co_mtp` | Cooperative (Improved) | Cross-agent attention fusion + multi-scale temporal encoding + deformable attention |

---

## Project Structure

```
collaborative_trajectory_prediction/
├── config.py                    # Global configuration (hyperparameters, paths, etc.)
├── setup.sh                     # One-click environment setup script
├── requirements.txt             # Python dependencies
├── quick_test.py                # Quick validation script (2 epochs)
├── run_experiments.py           # One-click full experiment runner
├── train.py                     # Unified training script
├── evaluate.py                  # Evaluation script
├── visualize.py                 # Visualization script
├── data/
│   ├── preprocess.py            # Data preprocessing
│   ├── generate_synthetic.py    # Synthetic data generation (for debugging)
│   └── download_v2x_seq.py      # Dataset download guide
├── datasets/
│   ├── trajectory_dataset.py    # PyTorch Dataset class
│   └── processed/               # Preprocessed data
├── models/
│   ├── __init__.py              # Model registry
│   ├── lstm_seq2seq.py          # LSTM Encoder-Decoder
│   ├── social_lstm.py           # Social LSTM
│   ├── grip_plus.py             # GRIP++
│   ├── transformer_pred.py      # Transformer Predictor
│   ├── v2x_graph.py             # V2X-Graph
│   ├── co_mtp.py                # Co-MTP
│   └── enhanced_co_mtp.py       # Enhanced Co-MTP (improved model)
├── utils/
│   ├── metrics.py               # Evaluation metrics (ADE/FDE/MR)
│   ├── losses.py                # Loss functions
│   └── helpers.py               # Utility functions
├── checkpoints/                 # Model weights
├── results/                     # Evaluation results (JSON)
└── visualizations/              # Visualization charts (PNG)
```

---

## Quick Start

### 1. Environment Setup

```bash
cd /data/yb/collaborative_trajectory_prediction
bash setup.sh
conda activate traj_pred
```

### 2. Data Preparation

**Dataset Source:** The V2X-Seq-TFD dataset used in this project is provided by the THU AIR Lab (Tsinghua University). Download link:

> [V2X-Seq-TFD Dataset (Google Drive)](https://drive.google.com/drive/folders/1yDnlrPCKImpVfI1OPBYyzLFWkhZP5v-7)

```bash
# Place V2X-Seq-TFD-Example.zip in the data/ directory
cd data
unzip V2X-Seq-TFD-Example.zip
cd ..

# Preprocess
python data/preprocess.py --data v2x_seq --mode cooperative
```

### 3. Quick Validation

```bash
python quick_test.py
```

### 4. Full Training

```bash
# Train all 7 models
python run_experiments.py

# Train a single model
python train.py --model enhanced_co_mtp

# Quick training (5 epochs, for debugging)
python run_experiments.py --quick
```

### 5. Evaluation and Visualization

```bash
python evaluate.py
python visualize.py
```

---

## Evaluation Metrics

| Metric | Full Name | Description |
|--------|-----------|-------------|
| **minADE** | Minimum Average Displacement Error | Average displacement error of the best predicted trajectory (meters) |
| **minFDE** | Minimum Final Displacement Error | Final point displacement error of the best predicted trajectory (meters) |
| **MR** | Miss Rate | Proportion of samples where the final point error exceeds the threshold |

Lower values indicate better performance for all metrics.

---

## Dataset Acknowledgment

The **V2X-Seq-TFD** dataset used in this project is provided by the THU AIR Lab (Tsinghua University). It is a large-scale sequential dataset for vehicle-infrastructure cooperative perception and forecasting.

| Item | Details |
|------|---------|
| **Dataset Name** | V2X-Seq-TFD |
| **Source** | THU AIR Lab, Tsinghua University |
| **Download** | [Google Drive](https://drive.google.com/drive/folders/1yDnlrPCKImpVfI1OPBYyzLFWkhZP5v-7) |
| **Original Paper** | Yu et al., "V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting," CVPR 2023 |
| **License** | For academic research purposes only. Please follow the original authors' license agreement and cite the original paper when using this dataset. |

---

## References

1. Hochreiter & Schmidhuber, "Long Short-Term Memory," Neural Computation, 1997.
2. Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces," CVPR 2016.
3. Li et al., "GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction," arXiv 2019.
4. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
5. Yu et al., "V2X-Seq: A Large-Scale Sequential Dataset for Vehicle-Infrastructure Cooperative Perception and Forecasting," CVPR 2023.
