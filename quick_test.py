"""
Quick test script to verify all models can train, evaluate and visualize.
Uses a tiny subset of data and 2 epochs for fast validation on CPU.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from datasets.trajectory_dataset import TrajectoryDataset, collate_fn
from models import get_model
from utils.metrics import compute_batch_metrics
from utils.losses import WinnerTakeAllLoss
from utils.helpers import set_seed, count_parameters, save_checkpoint

# Override config for quick testing
TEST_EPOCHS = 2
TEST_BATCH_SIZE = 16


def get_model_config(model_name):
    """Get configuration dict for a specific model with reduced sizes."""
    base_configs = {
        'lstm_seq2seq': LSTM_CONFIG.copy(),
        'social_lstm': SOCIAL_LSTM_CONFIG.copy(),
        'grip_plus': GRIP_PLUS_CONFIG.copy(),
        'transformer': TRANSFORMER_CONFIG.copy(),
        'v2x_graph': V2X_GRAPH_CONFIG.copy(),
        'co_mtp': CO_MTP_CONFIG.copy(),
        'enhanced_co_mtp': ENHANCED_CO_MTP_CONFIG.copy(),
    }
    # Reduce model sizes for quick CPU testing
    cfg = base_configs[model_name]
    if model_name == 'transformer':
        cfg['d_model'] = 64
        cfg['nhead'] = 4
        cfg['num_encoder_layers'] = 2
        cfg['num_decoder_layers'] = 2
        cfg['dim_feedforward'] = 128
    elif model_name == 'v2x_graph':
        cfg['hidden_dim'] = 64
        cfg['num_gnn_layers'] = 2
        cfg['num_heads'] = 2
    elif model_name in ('co_mtp', 'enhanced_co_mtp'):
        cfg['hidden_dim'] = 64
        cfg['d_model'] = 64
        cfg['nhead'] = 4
        cfg['num_gnn_layers'] = 2
        cfg['num_transformer_layers'] = 2
        if 'num_fusion_layers' in cfg:
            cfg['num_fusion_layers'] = 1
    elif model_name == 'lstm_seq2seq':
        cfg['hidden_dim'] = 64
        cfg['num_layers'] = 1
    elif model_name == 'social_lstm':
        cfg['hidden_dim'] = 64
        cfg['num_layers'] = 1
    elif model_name == 'grip_plus':
        cfg['hidden_dim'] = 64
        cfg['num_gcn_layers'] = 2
    # Reduce modes for speed
    cfg['num_modes'] = 3
    return cfg


def quick_train(model_name):
    """Quick train + eval for one model."""
    print(f"\n{'='*60}")
    print(f"Quick Test: {model_name}")
    print(f"{'='*60}")

    set_seed(SEED)
    device = torch.device('cpu')

    # Load a small subset of training data
    train_path = os.path.join(PROCESSED_DATA_DIR, 'train_samples.pt')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val_samples.pt')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test_samples.pt')

    train_data = torch.load(train_path, weights_only=False)[:200]
    val_data = torch.load(val_path, weights_only=False)[:100]
    test_data = torch.load(test_path, weights_only=False)[:100]

    # Save small subsets
    small_train = os.path.join(PROCESSED_DATA_DIR, 'small_train.pt')
    small_val = os.path.join(PROCESSED_DATA_DIR, 'small_val.pt')
    small_test = os.path.join(PROCESSED_DATA_DIR, 'small_test.pt')
    torch.save(train_data, small_train)
    torch.save(val_data, small_val)
    torch.save(test_data, small_test)

    train_dataset = TrajectoryDataset(small_train)
    val_dataset = TrajectoryDataset(small_val)
    test_dataset = TrajectoryDataset(small_test)

    train_loader = DataLoader(train_dataset, batch_size=TEST_BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=TEST_BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE,
                             shuffle=False, collate_fn=collate_fn)

    # Create model
    config = get_model_config(model_name)
    model = get_model(model_name, config).to(device)
    num_params = count_parameters(model)
    print(f"Model: {model.model_name}, Params: {num_params:,}, Cooperative: {model.cooperative}")

    criterion = WinnerTakeAllLoss(lambda_cls=1.0)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(1, TEST_EPOCHS + 1):
        model.train()
        start = time.time()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            batch_gpu = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_gpu[key] = val.to(device)
                else:
                    batch_gpu[key] = val

            output = model(batch_gpu)
            total_loss, _, _ = criterion(output['predictions'],
                                         output['mode_probs'],
                                         batch_gpu['future'])
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += total_loss.item()
            n_batches += 1

        # Validate
        model.eval()
        val_ade = 0
        val_fde = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch_gpu = {}
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor):
                        batch_gpu[key] = val.to(device)
                    else:
                        batch_gpu[key] = val
                output = model(batch_gpu)
                metrics = compute_batch_metrics(output['predictions'],
                                                batch_gpu['future'],
                                                output['predictions'].shape[1])
                bs = batch_gpu['future'].shape[0]
                val_ade += metrics['minADE'] * bs
                val_fde += metrics['minFDE'] * bs
                val_n += bs

        elapsed = time.time() - start
        avg_loss = train_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}/{TEST_EPOCHS} ({elapsed:.1f}s) | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Val minADE: {val_ade/max(val_n,1):.4f} | "
              f"Val minFDE: {val_fde/max(val_n,1):.4f}")

    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pt')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    avg_loss = train_loss / max(n_batches, 1)
    save_checkpoint(model, optimizer, TEST_EPOCHS, avg_loss, checkpoint_path)

    # Test evaluation
    model.eval()
    test_ade = 0
    test_fde = 0
    test_mr = 0
    test_n = 0
    all_preds = []
    all_gt = []
    all_hist = []

    with torch.no_grad():
        for batch in test_loader:
            batch_gpu = {}
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch_gpu[key] = val.to(device)
                else:
                    batch_gpu[key] = val
            output = model(batch_gpu)
            metrics = compute_batch_metrics(output['predictions'],
                                            batch_gpu['future'],
                                            output['predictions'].shape[1])
            bs = batch_gpu['future'].shape[0]
            test_ade += metrics['minADE'] * bs
            test_fde += metrics['minFDE'] * bs
            test_mr += metrics['MR'] * bs
            test_n += bs

            all_preds.append(output['predictions'].cpu())
            all_gt.append(batch_gpu['future'].cpu())
            all_hist.append(batch_gpu['history'].cpu())

    test_ade /= max(test_n, 1)
    test_fde /= max(test_n, 1)
    test_mr /= max(test_n, 1)

    print(f"\n  Test Results: minADE={test_ade:.4f}, minFDE={test_fde:.4f}, MR={test_mr:.4f}")

    # Save results
    results = {
        'model_name': model_name,
        'model_display_name': model.model_name,
        'cooperative': model.cooperative,
        'num_parameters': num_params,
        'best_epoch': TEST_EPOCHS,
        'test_metrics': {
            'minADE': test_ade,
            'minFDE': test_fde,
            'MR': test_mr,
            'loss': avg_loss,
        },
        'history': {
            'train': [{'loss': avg_loss}],
            'val': [{'loss': 0, 'minADE': val_ade/max(val_n,1),
                      'minFDE': val_fde/max(val_n,1), 'MR': 0}],
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f'{model_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Collect vis data
    vis_data = {
        'predictions': torch.cat(all_preds, dim=0),
        'ground_truth': torch.cat(all_gt, dim=0),
        'history': torch.cat(all_hist, dim=0),
    }

    return results, vis_data


def main():
    create_dirs()
    print("Quick Test: Verifying all models can train and evaluate")
    print(f"Using {TEST_EPOCHS} epochs, batch_size={TEST_BATCH_SIZE}, small data subset")

    # All 7 models
    model_names = [
        'lstm_seq2seq',
        'social_lstm',
        'grip_plus',
        'transformer',
        'v2x_graph',
        'co_mtp',
        'enhanced_co_mtp',
    ]
    all_results = {}
    all_vis_data = {}

    for model_name in model_names:
        try:
            results, vis_data = quick_train(model_name)
            all_results[model_name] = results
            all_vis_data[model_name] = vis_data
            print(f"  [PASS] {model_name}")
        except Exception as e:
            print(f"  [FAIL] {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save combined results
    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save vis data
    torch.save(all_vis_data, os.path.join(RESULTS_DIR, 'visualization_data.pt'))

    # Print summary
    print(f"\n{'='*90}")
    print("QUICK TEST RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<25} {'Cooperative':<12} {'Params':<12} "
          f"{'minADE':<10} {'minFDE':<10} {'MR':<10}")
    print("-" * 90)
    for name, res in all_results.items():
        tm = res['test_metrics']
        print(f"{res['model_display_name']:<25} "
              f"{'Yes' if res['cooperative'] else 'No':<12} "
              f"{res['num_parameters']:>10,} "
              f"{tm['minADE']:>9.4f} "
              f"{tm['minFDE']:>9.4f} "
              f"{tm['MR']:>9.4f}")
    print("=" * 90)

    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        from visualize import (plot_trajectory_samples, plot_metrics_comparison,
                               plot_model_comparison_trajectories,
                               plot_cooperative_vs_noncooperative)

        for model_name, vis_data in all_vis_data.items():
            plot_trajectory_samples(vis_data, model_name)

        if len(all_vis_data) > 1:
            plot_model_comparison_trajectories(all_vis_data)

        plot_metrics_comparison()
        plot_cooperative_vs_noncooperative()
        print("Visualizations generated successfully!")
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nAll done! Check results/ and visualizations/ directories.")


if __name__ == '__main__':
    main()
