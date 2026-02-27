"""
Evaluation script for trained trajectory prediction models.

Loads saved checkpoints and evaluates on the test set.
Generates detailed metrics comparison across all models.

Usage:
    python evaluate.py                    # evaluate all trained models
    python evaluate.py --model lstm_seq2seq  # evaluate specific model
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from datasets.trajectory_dataset import get_dataloaders
from models import get_model
from utils.metrics import compute_min_ade, compute_min_fde, compute_miss_rate
from utils.losses import WinnerTakeAllLoss
from utils.helpers import set_seed, load_checkpoint


def get_model_config(model_name):
    """Get configuration dict for a specific model."""
    configs = {
        'lstm_seq2seq': LSTM_CONFIG,
        'social_lstm': SOCIAL_LSTM_CONFIG,
        'grip_plus': GRIP_PLUS_CONFIG,
        'transformer': TRANSFORMER_CONFIG,
        'v2x_graph': V2X_GRAPH_CONFIG,
        'co_mtp': CO_MTP_CONFIG,
        'enhanced_co_mtp': ENHANCED_CO_MTP_CONFIG,
    }
    return configs[model_name]


@torch.no_grad()
def detailed_evaluation(model, test_loader, device):
    """
    Perform detailed evaluation with per-sample metrics.

    Returns:
        dict with aggregated and per-sample metrics
    """
    model.eval()

    all_ade = []
    all_fde = []
    all_mr = []
    all_predictions = []
    all_gt = []
    all_histories = []

    for batch in test_loader:
        batch_gpu = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch_gpu[key] = val.to(device)
            else:
                batch_gpu[key] = val

        output = model(batch_gpu)
        predictions = output['predictions']  # (B, K, T_f, 2)
        gt = batch_gpu['future']  # (B, T_f, 2)
        history = batch_gpu['history']  # (B, T_h, 2)

        B = gt.shape[0]
        for i in range(B):
            pred_i = predictions[i]  # (K, T_f, 2)
            gt_i = gt[i]  # (T_f, 2)

            ade = compute_min_ade(pred_i, gt_i)
            fde = compute_min_fde(pred_i, gt_i)
            mr = compute_miss_rate(pred_i, gt_i, threshold=MISS_RATE_THRESHOLD)

            all_ade.append(ade)
            all_fde.append(fde)
            all_mr.append(mr)

        all_predictions.append(predictions.cpu())
        all_gt.append(gt.cpu())
        all_histories.append(history.cpu())

    # Aggregate
    all_ade = np.array(all_ade)
    all_fde = np.array(all_fde)
    all_mr = np.array(all_mr)

    results = {
        'minADE_mean': float(np.mean(all_ade)),
        'minADE_std': float(np.std(all_ade)),
        'minADE_median': float(np.median(all_ade)),
        'minFDE_mean': float(np.mean(all_fde)),
        'minFDE_std': float(np.std(all_fde)),
        'minFDE_median': float(np.median(all_fde)),
        'MR': float(np.mean(all_mr)),
        'num_samples': len(all_ade),
        'per_sample_ade': all_ade.tolist(),
        'per_sample_fde': all_fde.tolist(),
    }

    # Percentile analysis
    for p in [50, 75, 90, 95, 99]:
        results[f'ADE_p{p}'] = float(np.percentile(all_ade, p))
        results[f'FDE_p{p}'] = float(np.percentile(all_fde, p))

    # Collect tensors for visualization
    all_predictions = torch.cat(all_predictions, dim=0)
    all_gt = torch.cat(all_gt, dim=0)
    all_histories = torch.cat(all_histories, dim=0)

    return results, all_predictions, all_gt, all_histories


def evaluate_model(model_name, device):
    """Evaluate a single model."""
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found for {model_name}: {checkpoint_path}")
        return None

    config = get_model_config(model_name)
    model = get_model(model_name, config).to(device)

    # Load checkpoint
    load_checkpoint(model, None, checkpoint_path, device=str(device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate trajectory prediction models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lstm_seq2seq', 'transformer', 'v2x_graph',
                                 'co_mtp', 'all'])
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    set_seed(SEED)
    device = DEVICE
    print(f"Device: {device}")

    # Load test data
    _, _, test_loader = get_dataloaders(batch_size=args.batch_size)

    if args.model == 'all':
        model_names = ['lstm_seq2seq', 'transformer', 'v2x_graph', 'co_mtp']
    else:
        model_names = [args.model]

    all_results = {}
    all_vis_data = {}

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        model = evaluate_model(model_name, device)
        if model is None:
            continue

        results, preds, gt, hist = detailed_evaluation(model, test_loader, device)
        all_results[model_name] = results
        all_vis_data[model_name] = {
            'predictions': preds,
            'ground_truth': gt,
            'history': hist,
        }

        print(f"\n  minADE: {results['minADE_mean']:.4f} ± {results['minADE_std']:.4f}")
        print(f"  minFDE: {results['minFDE_mean']:.4f} ± {results['minFDE_std']:.4f}")
        print(f"  MR@{MISS_RATE_THRESHOLD}m: {results['MR']:.4f}")
        print(f"  Samples: {results['num_samples']}")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    eval_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation results saved to: {eval_path}")

    # Save visualization data
    vis_data_path = os.path.join(RESULTS_DIR, 'visualization_data.pt')
    torch.save(all_vis_data, vis_data_path)
    print(f"Visualization data saved to: {vis_data_path}")

    # Print comparison table
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"{'Model':<20} {'minADE':<12} {'minFDE':<12} "
              f"{'MR':<10} {'ADE_p90':<12} {'FDE_p90':<12}")
        print("-" * 80)
        for name, res in all_results.items():
            print(f"{name:<20} "
                  f"{res['minADE_mean']:.4f}±{res['minADE_std']:.3f} "
                  f"{res['minFDE_mean']:.4f}±{res['minFDE_std']:.3f} "
                  f"{res['MR']:.4f}    "
                  f"{res['ADE_p90']:.4f}      "
                  f"{res['FDE_p90']:.4f}")
        print("=" * 80)


if __name__ == '__main__':
    main()
