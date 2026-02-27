#!/bin/bash
# ============================================================
# 服务器环境一键部署脚本
# 用法: cd /data/yb/collaborative_trajectory_prediction && bash setup.sh
# ============================================================

set -e

echo "============================================================"
echo "  协同车辆轨迹预测 - 环境部署脚本"
echo "============================================================"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Project directory: $PROJECT_DIR"

# [1] 检查是否有conda
echo ""
echo "[1/5] 检查Conda环境..."
USE_CONDA=false
if command -v conda &> /dev/null; then
    echo "  Conda found: $(conda --version)"
    USE_CONDA=true
else
    echo "  Conda not found, trying to install Miniconda..."
    if [ ! -f /tmp/miniconda.sh ]; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    fi
    bash /tmp/miniconda.sh -b -p $HOME/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    source ~/.bashrc
    USE_CONDA=true
    echo "  Miniconda installed successfully"
fi

# [2] 创建conda环境
echo ""
echo "[2/5] 创建Conda虚拟环境 traj_pred..."
if conda env list | grep -q "traj_pred"; then
    echo "  Environment traj_pred already exists"
else
    conda create -n traj_pred python=3.10 -y
    echo "  Environment traj_pred created"
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate traj_pred
echo "  Activated: $(which python)"

# [3] 检查CUDA/GPU
echo ""
echo "[3/5] 检查GPU/CUDA..."
HAS_GPU=false
CUDA_VER=""
if command -v nvidia-smi &> /dev/null; then
    echo "  NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "  CUDA Version: $CUDA_VER"
    HAS_GPU=true
else
    echo "  No NVIDIA GPU detected, will use CPU"
fi

# [4] 安装PyTorch
echo ""
echo "[4/5] 安装PyTorch..."
if [ "$HAS_GPU" = true ]; then
    echo "  Installing PyTorch with CUDA support..."
    MAJOR=$(echo $CUDA_VER | cut -d. -f1)
    if [ "$MAJOR" = "12" ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
    elif [ "$MAJOR" = "11" ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -3
    else
        pip install torch torchvision 2>&1 | tail -3
    fi
else
    echo "  Installing PyTorch (CPU only)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
fi

# [5] 安装其他依赖
echo ""
echo "[5/5] 安装其他依赖..."
pip install -r "$PROJECT_DIR/requirements.txt" 2>&1 | tail -5

# 安装 gdown（用于从Google Drive下载数据集）
pip install gdown 2>&1 | tail -2

# 安装 p7zip（用于解压分卷压缩包）
if command -v apt-get &> /dev/null; then
    sudo apt-get install -y p7zip-full 2>/dev/null || echo "  p7zip install skipped (no sudo)"
fi

# 验证安装
echo ""
echo "验证安装..."
python -c "
import torch
print(f'  PyTorch version: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
import numpy, matplotlib, tqdm, sklearn
print('  All dependencies OK!')
"

# 创建目录
mkdir -p "$PROJECT_DIR/data/V2X-Seq-TFD-Example"
mkdir -p "$PROJECT_DIR/data/V2X-Seq-TFD"
mkdir -p "$PROJECT_DIR/datasets/processed"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/visualizations"

echo ""
echo "============================================================"
echo "  部署完成！"
echo "============================================================"
echo ""
echo "后续步骤："
echo ""
echo "1. 每次登录服务器后激活环境："
echo "   conda activate traj_pred"
echo "   cd $PROJECT_DIR"
echo ""
echo "2. 放置数据集（解压V2X-Seq-TFD-Example.zip到data/目录）："
echo "   cd $PROJECT_DIR/data"
echo "   unzip /path/to/V2X-Seq-TFD-Example.zip"
echo ""
echo "3. 预处理数据："
echo "   python data/preprocess.py --data v2x_seq --mode cooperative"
echo ""
echo "4. 快速验证（约1分钟）："
echo "   python quick_test.py"
echo ""
echo "5. 完整训练所有7个模型："
echo "   python run_experiments.py"
echo ""
echo "6. 单独训练某个模型："
echo "   python train.py --model lstm_seq2seq"
echo "   python train.py --model social_lstm"
echo "   python train.py --model grip_plus"
echo "   python train.py --model transformer"
echo "   python train.py --model v2x_graph"
echo "   python train.py --model co_mtp"
echo "   python train.py --model enhanced_co_mtp"
echo ""
echo "7. 后台训练（防止断连）："
echo "   tmux new -s train"
echo "   conda activate traj_pred"
echo "   python run_experiments.py"
echo "   # 按 Ctrl+B 然后按 D 脱离"
echo "   # 重新连接: tmux attach -t train"
echo ""
echo "============================================================"
