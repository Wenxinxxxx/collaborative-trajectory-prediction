"""
Visualization script for trajectory prediction results.

Generates:
  1. Trajectory prediction plots (history + GT + predicted modes)
  2. Training curves (loss, ADE, FDE over epochs)
  3. Model comparison bar charts
  4. Error distribution histograms
  5. Qualitative comparison across models

Usage:
    python visualize.py                  # generate all visualizations
    python visualize.py --type trajectory  # only trajectory plots
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

# Style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': VIS_DPI,
})
sns.set_style("whitegrid")

# Color palette
COLORS = {
    'history': '#2196F3',       # Blue
    'ground_truth': '#4CAF50',  # Green
    'prediction': '#FF5722',    # Red/Orange
    'mode_colors': ['#FF5722', '#FF9800', '#FFC107', '#9C27B0', '#00BCD4', '#795548'],
}

MODEL_COLORS = {
    'lstm_seq2seq': '#2196F3',
    'transformer': '#FF9800',
    'v2x_graph': '#4CAF50',
    'co_mtp': '#E91E63',
}

MODEL_LABELS = {
    'lstm_seq2seq': 'LSTM Seq2Seq',
    'transformer': 'Transformer',
    'v2x_graph': 'V2X-Graph',
    'co_mtp': 'Co-MTP',
}


def plot_trajectory_samples(vis_data, model_name, num_samples=6, save_dir=None):
    """
    Plot trajectory prediction samples for a single model.

    Shows history (blue), ground truth (green), and predicted modes (red shades).
    """
    if save_dir is None:
        save_dir = VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    preds = vis_data['predictions']  # (N, K, T_f, 2)
    gt = vis_data['ground_truth']    # (N, T_f, 2)
    hist = vis_data['history']       # (N, T_h, 2)

    num_samples = min(num_samples, len(gt))
    indices = np.linspace(0, len(gt) - 1, num_samples).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Trajectory Predictions - {MODEL_LABELS.get(model_name, model_name)}',
                 fontsize=16, fontweight='bold')

    for idx, (ax, sample_idx) in enumerate(zip(axes.flatten(), indices)):
        h = hist[sample_idx].numpy()
        g = gt[sample_idx].numpy()
        p = preds[sample_idx].numpy()  # (K, T_f, 2)

        # Plot history
        ax.plot(h[:, 0], h[:, 1], 'o-', color=COLORS['history'],
                markersize=3, linewidth=2, label='History', zorder=3)

        # Plot ground truth
        ax.plot(g[:, 0], g[:, 1], 's-', color=COLORS['ground_truth'],
                markersize=3, linewidth=2, label='Ground Truth', zorder=3)

        # Plot predicted modes
        K = p.shape[0]
        for k in range(K):
            color = COLORS['mode_colors'][k % len(COLORS['mode_colors'])]
            alpha = 0.8 if k == 0 else 0.4
            label = f'Mode {k+1}' if k < 3 else None
            ax.plot(p[k, :, 0], p[k, :, 1], '--', color=color,
                    alpha=alpha, linewidth=1.5, label=label, zorder=2)

        # Mark start and end points
        ax.plot(h[0, 0], h[0, 1], 'D', color=COLORS['history'],
                markersize=8, zorder=4)
        ax.plot(g[-1, 0], g[-1, 1], '*', color=COLORS['ground_truth'],
                markersize=12, zorder=4)

        ax.set_title(f'Sample {sample_idx}')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        if idx == 0:
            ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'trajectories_{model_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison_trajectories(all_vis_data, sample_idx=0, save_dir=None):
    """
    Compare trajectory predictions from all models on the same sample.
    """
    if save_dir is None:
        save_dir = VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    num_models = len(all_vis_data)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    if num_models == 1:
        axes = [axes]

    fig.suptitle(f'Model Comparison - Sample {sample_idx}',
                 fontsize=16, fontweight='bold')

    for ax, (model_name, vis_data) in zip(axes, all_vis_data.items()):
        hist = vis_data['history'][sample_idx].numpy()
        gt = vis_data['ground_truth'][sample_idx].numpy()
        preds = vis_data['predictions'][sample_idx].numpy()

        # History
        ax.plot(hist[:, 0], hist[:, 1], 'o-', color=COLORS['history'],
                markersize=3, linewidth=2, label='History')
        # Ground truth
        ax.plot(gt[:, 0], gt[:, 1], 's-', color=COLORS['ground_truth'],
                markersize=3, linewidth=2, label='Ground Truth')
        # Best prediction (mode 0)
        ax.plot(preds[0, :, 0], preds[0, :, 1], '--',
                color=MODEL_COLORS.get(model_name, '#FF5722'),
                linewidth=2, label='Best Prediction')
        # Other modes
        for k in range(1, min(3, preds.shape[0])):
            ax.plot(preds[k, :, 0], preds[k, :, 1], '--',
                    color=MODEL_COLORS.get(model_name, '#FF5722'),
                    alpha=0.3, linewidth=1)

        ax.set_title(MODEL_LABELS.get(model_name, model_name))
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.legend(loc='best', fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'model_comparison_trajectories.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(save_dir=None):
    """Plot training loss and validation metrics curves."""
    if save_dir is None:
        save_dir = VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')

    model_names = ['lstm_seq2seq', 'transformer', 'v2x_graph', 'co_mtp']

    for model_name in model_names:
        results_path = os.path.join(RESULTS_DIR, f'{model_name}_results.json')
        if not os.path.exists(results_path):
            continue

        with open(results_path, 'r') as f:
            results = json.load(f)

        history = results['history']
        epochs = range(1, len(history['train']) + 1)
        color = MODEL_COLORS.get(model_name, '#000000')
        label = MODEL_LABELS.get(model_name, model_name)

        # Training loss
        train_losses = [h['loss'] for h in history['train']]
        axes[0, 0].plot(epochs, train_losses, '-', color=color, label=label)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')

        # Validation loss
        val_losses = [h['loss'] for h in history['val']]
        axes[0, 1].plot(epochs, val_losses, '-', color=color, label=label)
        axes[0, 1].set_title('Validation Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')

        # Validation minADE
        val_ade = [h['minADE'] for h in history['val']]
        axes[1, 0].plot(epochs, val_ade, '-', color=color, label=label)
        axes[1, 0].set_title('Validation minADE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('minADE (m)')

        # Validation minFDE
        val_fde = [h['minFDE'] for h in history['val']]
        axes[1, 1].plot(epochs, val_fde, '-', color=color, label=label)
        axes[1, 1].set_title('Validation minFDE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('minFDE (m)')

    for ax in axes.flatten():
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(save_dir=None):
    """Plot bar chart comparing metrics across models."""
    if save_dir is None:
        save_dir = VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    eval_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    if not os.path.exists(eval_path):
        # Try to build from individual results
        all_results = {}
        for model_name in ['lstm_seq2seq', 'transformer', 'v2x_graph', 'co_mtp']:
            rp = os.path.join(RESULTS_DIR, f'{model_name}_results.json')
            if os.path.exists(rp):
                with open(rp, 'r') as f:
                    data = json.load(f)
                all_results[model_name] = data['test_metrics']
    else:
        with open(eval_path, 'r') as f:
            all_results = json.load(f)

    if not all_results:
        print("No evaluation results found.")
        return

    models = list(all_results.keys())
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, '#888888') for m in models]

    metrics = ['minADE', 'minFDE', 'MR']
    metric_labels = ['minADE (m)', 'minFDE (m)', 'Miss Rate']

    # Handle both key formats
    def get_metric(res, metric):
        if f'{metric}_mean' in res:
            return res[f'{metric}_mean']
        return res.get(metric, 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        values = [get_metric(all_results[m], metric) for m in models]
        bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=1.5)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10)

        ax.set_title(metric_label)
        ax.set_ylabel(metric_label)
        ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_error_distributions(save_dir=None):
    """Plot error distribution histograms for each model."""
    if save_dir is None:
        save_dir = VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    eval_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    if not os.path.exists(eval_path):
        print("No evaluation results with per-sample data found.")
        return

    with open(eval_path, 'r') as f:
        all_results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Error Distributions', fontsize=16, fontweight='bold')

    for model_name, results in all_results.items():
        if 'per_sample_ade' not in results:
            continue
        color = MODEL_COLORS.get(model_name, '#888888')
        label = MODEL_LABELS.get(model_name, model_name)

        ade_data = np.array(results['per_sample_ade'])
        fde_data = np.array(results['per_sample_fde'])

        axes[0].hist(ade_data, bins=50, alpha=0.5, color=color,
                     label=label, density=True)
        axes[1].hist(fde_data, bins=50, alpha=0.5, color=color,
                     label=label, density=True)

    axes[0].set_title('minADE Distribution')
    axes[0].set_xlabel('minADE (m)')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    axes[1].set_title('minFDE Distribution')
    axes[1].set_xlabel('minFDE (m)')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'error_distributions.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cooperative_vs_noncooperative(save_dir=None):
    """
    Plot comparison between cooperative and non-cooperative models.
    Highlights the benefit of V2X cooperation.
    """
    if save_dir is None:
        save_dir = VIS_DIR
    os.makedirs(save_dir, exist_ok=True)

    # Load results
    all_results = {}
    for model_name in ['lstm_seq2seq', 'transformer', 'v2x_graph', 'co_mtp']:
        rp = os.path.join(RESULTS_DIR, f'{model_name}_results.json')
        if os.path.exists(rp):
            with open(rp, 'r') as f:
                all_results[model_name] = json.load(f)

    if len(all_results) < 2:
        print("Not enough models for cooperative comparison.")
        return

    # Separate cooperative and non-cooperative
    non_coop = {k: v for k, v in all_results.items() if not v.get('cooperative', False)}
    coop = {k: v for k, v in all_results.items() if v.get('cooperative', False)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cooperative vs Non-Cooperative Prediction',
                 fontsize=16, fontweight='bold')

    # minADE comparison
    ax = axes[0]
    x_pos = 0
    for name, data in non_coop.items():
        tm = data['test_metrics']
        ade = tm.get('minADE', tm.get('minADE_mean', 0))
        ax.bar(x_pos, ade, color='#90CAF9', edgecolor='white',
               width=0.8, label='Non-Cooperative' if x_pos == 0 else None)
        ax.text(x_pos, ade + 0.01, f'{ade:.3f}', ha='center', fontsize=9)
        x_pos += 1

    for name, data in coop.items():
        tm = data['test_metrics']
        ade = tm.get('minADE', tm.get('minADE_mean', 0))
        ax.bar(x_pos, ade, color='#A5D6A7', edgecolor='white',
               width=0.8, label='Cooperative' if x_pos == len(non_coop) else None)
        ax.text(x_pos, ade + 0.01, f'{ade:.3f}', ha='center', fontsize=9)
        x_pos += 1

    all_names = list(non_coop.keys()) + list(coop.keys())
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels([MODEL_LABELS.get(n, n) for n in all_names], rotation=15)
    ax.set_title('minADE Comparison')
    ax.set_ylabel('minADE (m)')
    ax.legend()

    # minFDE comparison
    ax = axes[1]
    x_pos = 0
    for name, data in non_coop.items():
        tm = data['test_metrics']
        fde = tm.get('minFDE', tm.get('minFDE_mean', 0))
        ax.bar(x_pos, fde, color='#90CAF9', edgecolor='white',
               width=0.8, label='Non-Cooperative' if x_pos == 0 else None)
        ax.text(x_pos, fde + 0.01, f'{fde:.3f}', ha='center', fontsize=9)
        x_pos += 1

    for name, data in coop.items():
        tm = data['test_metrics']
        fde = tm.get('minFDE', tm.get('minFDE_mean', 0))
        ax.bar(x_pos, fde, color='#A5D6A7', edgecolor='white',
               width=0.8, label='Cooperative' if x_pos == len(non_coop) else None)
        ax.text(x_pos, fde + 0.01, f'{fde:.3f}', ha='center', fontsize=9)
        x_pos += 1

    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels([MODEL_LABELS.get(n, n) for n in all_names], rotation=15)
    ax.set_title('minFDE Comparison')
    ax.set_ylabel('minFDE (m)')
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cooperative_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations')
    parser.add_argument('--type', type=str, default='all',
                        choices=['all', 'trajectory', 'training', 'comparison',
                                 'distribution', 'cooperative'])
    args = parser.parse_args()

    os.makedirs(VIS_DIR, exist_ok=True)

    # Load visualization data if available
    vis_data_path = os.path.join(RESULTS_DIR, 'visualization_data.pt')
    all_vis_data = None
    if os.path.exists(vis_data_path):
        all_vis_data = torch.load(vis_data_path, weights_only=False)

    if args.type in ['all', 'trajectory'] and all_vis_data:
        for model_name, vis_data in all_vis_data.items():
            plot_trajectory_samples(vis_data, model_name)
        if len(all_vis_data) > 1:
            plot_model_comparison_trajectories(all_vis_data)

    if args.type in ['all', 'training']:
        plot_training_curves()

    if args.type in ['all', 'comparison']:
        plot_metrics_comparison()

    if args.type in ['all', 'distribution']:
        plot_error_distributions()

    if args.type in ['all', 'cooperative']:
        plot_cooperative_vs_noncooperative()

    print(f"\nAll visualizations saved to: {VIS_DIR}")


if __name__ == '__main__':
    main()
