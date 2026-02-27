"""
Configuration file for Collaborative Vehicle Trajectory Prediction.
All hyperparameters and paths are centralized here for easy modification.
"""

import os
import torch

# ============================================================
# Path Configuration
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'processed')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
VIS_DIR = os.path.join(PROJECT_ROOT, 'visualizations')

# V2X-Seq TFD dataset root
# For Example data:  data/V2X-Seq-TFD-Example
# For Full data:     data/V2X-Seq-TFD
V2X_SEQ_ROOT = os.path.join(RAW_DATA_DIR, 'V2X-Seq-TFD-Example')

# ============================================================
# Data Configuration
# ============================================================
DATA_FREQ = 10  # V2X-Seq is 10Hz (0.1s per step)

# Each scenario has 100 timesteps = 10 seconds
# We use first 50 steps (5s) as history, last 50 steps (5s) as future
HISTORY_STEPS = 50   # 5 seconds of history at 10Hz
FUTURE_STEPS = 50    # 5 seconds of future at 10Hz

# Input feature dimension
INPUT_DIM = 2  # (x, y) coordinates

# Number of nearby agents to consider for interaction modeling
MAX_AGENTS = 16

# Map feature dimension
MAP_DIM = 2  # (x, y) for lane centerline points
MAX_LANES = 20
LANE_POINTS = 20  # number of sampled points per lane

# Number of modalities for multi-modal prediction
NUM_MODES = 6

# ============================================================
# Training Configuration
# ============================================================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5

# V2X-Seq TFD already has train/val split, no need for random split
SEED = 42

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Model-Specific Configurations
# ============================================================

# --- LSTM Seq2Seq ---
LSTM_CONFIG = {
    'input_dim': INPUT_DIM,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'output_dim': INPUT_DIM,
    'future_steps': FUTURE_STEPS,
    'num_modes': NUM_MODES,
}

# --- Transformer Predictor ---
TRANSFORMER_CONFIG = {
    'input_dim': INPUT_DIM,
    'd_model': 128,
    'nhead': 8,
    'num_encoder_layers': 4,
    'num_decoder_layers': 4,
    'dim_feedforward': 256,
    'dropout': 0.1,
    'output_dim': INPUT_DIM,
    'history_steps': HISTORY_STEPS,
    'future_steps': FUTURE_STEPS,
    'num_modes': NUM_MODES,
}

# --- V2X-Graph ---
V2X_GRAPH_CONFIG = {
    'input_dim': INPUT_DIM,
    'hidden_dim': 128,
    'num_gnn_layers': 3,
    'num_heads': 4,
    'dropout': 0.1,
    'output_dim': INPUT_DIM,
    'history_steps': HISTORY_STEPS,
    'future_steps': FUTURE_STEPS,
    'max_agents': MAX_AGENTS,
    'map_dim': MAP_DIM,
    'max_lanes': MAX_LANES,
    'lane_points': LANE_POINTS,
    'num_modes': NUM_MODES,
}

# --- Social LSTM ---
SOCIAL_LSTM_CONFIG = {
    'input_dim': INPUT_DIM,
    'hidden_dim': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'output_dim': INPUT_DIM,
    'future_steps': FUTURE_STEPS,
    'num_modes': NUM_MODES,
    'max_agents': MAX_AGENTS,
}

# --- GRIP++ ---
GRIP_PLUS_CONFIG = {
    'input_dim': INPUT_DIM,
    'hidden_dim': 128,
    'num_gcn_layers': 3,
    'dropout': 0.1,
    'output_dim': INPUT_DIM,
    'history_steps': HISTORY_STEPS,
    'future_steps': FUTURE_STEPS,
    'max_agents': MAX_AGENTS,
    'num_modes': NUM_MODES,
}

# --- Co-MTP ---
CO_MTP_CONFIG = {
    'input_dim': INPUT_DIM,
    'hidden_dim': 128,
    'd_model': 128,
    'nhead': 8,
    'num_gnn_layers': 3,
    'num_transformer_layers': 3,
    'dropout': 0.1,
    'output_dim': INPUT_DIM,
    'history_steps': HISTORY_STEPS,
    'future_steps': FUTURE_STEPS,
    'max_agents': MAX_AGENTS,
    'map_dim': MAP_DIM,
    'max_lanes': MAX_LANES,
    'lane_points': LANE_POINTS,
    'num_modes': NUM_MODES,
    'num_temporal_scales': 3,
}

# --- Enhanced Co-MTP (Proposed Improvement) ---
ENHANCED_CO_MTP_CONFIG = {
    'input_dim': INPUT_DIM,
    'hidden_dim': 128,
    'd_model': 128,
    'nhead': 8,
    'num_gnn_layers': 3,
    'num_transformer_layers': 3,
    'num_fusion_layers': 2,
    'dropout': 0.1,
    'output_dim': INPUT_DIM,
    'history_steps': HISTORY_STEPS,
    'future_steps': FUTURE_STEPS,
    'max_agents': MAX_AGENTS,
    'map_dim': MAP_DIM,
    'max_lanes': MAX_LANES,
    'lane_points': LANE_POINTS,
    'num_modes': NUM_MODES,
    'num_temporal_scales': 3,
}

# ============================================================
# Evaluation Configuration
# ============================================================
EVAL_METRICS = ['minADE', 'minFDE', 'MR']
MISS_RATE_THRESHOLD = 2.0  # meters

# ============================================================
# Visualization Configuration
# ============================================================
VIS_NUM_SAMPLES = 10
VIS_DPI = 150
VIS_FIGSIZE = (10, 8)


def create_dirs():
    """Create all necessary directories."""
    for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR,
              RESULTS_DIR, VIS_DIR]:
        os.makedirs(d, exist_ok=True)


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"V2X-Seq Root: {V2X_SEQ_ROOT}")
    print(f"History: {HISTORY_STEPS} steps ({HISTORY_STEPS/DATA_FREQ:.1f}s)")
    print(f"Future:  {FUTURE_STEPS} steps ({FUTURE_STEPS/DATA_FREQ:.1f}s)")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Prediction Modes: {NUM_MODES}")
    print(f"Max Agents: {MAX_AGENTS}")
    print("=" * 60)


if __name__ == '__main__':
    create_dirs()
    print_config()
