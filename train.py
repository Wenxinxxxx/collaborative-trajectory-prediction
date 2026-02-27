"""
Unified training script for all trajectory prediction models.

Usage:
    python train.py --model lstm_seq2seq
    python train.py --model transformer
    python train.py --model v2x_graph
    python train.py --model co_mtp
    python train.py --model all   # train all models sequentially
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
from datasets.trajectory_dataset import get_dataloaders
from models import get_model
from utils.metrics import compute_batch_metrics
from utils.losses import WinnerTakeAllLoss
from utils.helpers import (set_seed, count_parameters, save_checkpoint,
                           EarlyStopping, AverageMeter)


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


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    loss_meter = AverageMeter()
    reg_meter = AverageMeter()
    cls_meter = AverageMeter()

    for batch_idx, batch in enumerate(loader):
        # Move to device
        batch_gpu = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch_gpu[key] = val.to(device)
            else:
                batch_gpu[key] = val

        # Forward pass
        output = model(batch_gpu)
        predictions = output['predictions']
        mode_probs = output['mode_probs']
        gt = batch_gpu['future']

        # Compute loss
        total_loss, reg_loss, cls_loss = criterion(predictions, mode_probs, gt)

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Update meters
        bs = gt.shape[0]
        loss_meter.update(total_loss.item(), bs)
        reg_meter.update(reg_loss.item(), bs)
        cls_meter.update(cls_loss.item(), bs)

    return {
        'loss': loss_meter.avg,
        'reg_loss': reg_meter.avg,
        'cls_loss': cls_meter.avg,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    loss_meter = AverageMeter()
    ade_meter = AverageMeter()
    fde_meter = AverageMeter()
    mr_meter = AverageMeter()

    for batch in loader:
        batch_gpu = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch_gpu[key] = val.to(device)
            else:
                batch_gpu[key] = val

        output = model(batch_gpu)
        predictions = output['predictions']
        mode_probs = output['mode_probs']
        gt = batch_gpu['future']

        # Loss
        total_loss, _, _ = criterion(predictions, mode_probs, gt)
        loss_meter.update(total_loss.item(), gt.shape[0])

        # Metrics
        metrics = compute_batch_metrics(predictions, gt, predictions.shape[1])
        ade_meter.update(metrics['minADE'], gt.shape[0])
        fde_meter.update(metrics['minFDE'], gt.shape[0])
        mr_meter.update(metrics['MR'], gt.shape[0])

    return {
        'loss': loss_meter.avg,
        'minADE': ade_meter.avg,
        'minFDE': fde_meter.avg,
        'MR': mr_meter.avg,
    }


def train_model(model_name, num_epochs=None, batch_size=None):
    """
    Full training pipeline for a single model.

    Args:
        model_name: one of 'lstm_seq2seq', 'transformer', 'v2x_graph', 'co_mtp'
        num_epochs: override default epochs
        batch_size: override default batch size
    """
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE

    print("=" * 70)
    print(f"Training: {model_name}")
    print("=" * 70)

    # Setup
    set_seed(SEED)
    device = DEVICE
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, num_workers=0
    )

    # Model
    config = get_model_config(model_name)
    model = get_model(model_name, config).to(device)
    num_params = count_parameters(model)
    print(f"Model: {model.model_name}")
    print(f"Parameters: {num_params:,}")
    print(f"Cooperative: {model.cooperative}")

    # Loss, optimizer, scheduler
    criterion = WinnerTakeAllLoss(lambda_cls=1.0)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                     weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=SCHEDULER_FACTOR,
                                  patience=SCHEDULER_PATIENCE)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # Training loop
    best_val_loss = float('inf')
    history = {'train': [], 'val': []}

    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{model_name}_best.pt')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_stats = train_one_epoch(model, train_loader, optimizer,
                                      criterion, device)

        # Validate
        val_stats = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_stats['loss'])

        # Record history
        history['train'].append(train_stats)
        history['val'].append(val_stats)

        elapsed = time.time() - start_time

        # Print progress
        print(f"Epoch {epoch:3d}/{num_epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_stats['loss']:.4f} | "
              f"Val Loss: {val_stats['loss']:.4f} | "
              f"minADE: {val_stats['minADE']:.4f} | "
              f"minFDE: {val_stats['minFDE']:.4f} | "
              f"MR: {val_stats['MR']:.4f}")

        # Save best model
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            checkpoint_path)
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")

        # Early stopping
        early_stopping(val_stats['loss'])
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Final test evaluation
    print("\n" + "-" * 50)
    print("Final Test Evaluation")
    print("-" * 50)

    # Load best model
    from utils.helpers import load_checkpoint
    load_checkpoint(model, None, checkpoint_path, device=str(device))
    test_stats = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss:  {test_stats['loss']:.4f}")
    print(f"Test minADE: {test_stats['minADE']:.4f}")
    print(f"Test minFDE: {test_stats['minFDE']:.4f}")
    print(f"Test MR:     {test_stats['MR']:.4f}")

    # Save results
    results = {
        'model_name': model_name,
        'model_display_name': model.model_name,
        'cooperative': model.cooperative,
        'num_parameters': num_params,
        'best_epoch': epoch,
        'test_metrics': test_stats,
        'history': history,
    }

    results_path = os.path.join(RESULTS_DIR, f'{model_name}_results.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train trajectory prediction models')
    parser.add_argument('--model', type=str, default='all',
                        choices=['lstm_seq2seq', 'transformer', 'v2x_graph',
                                 'co_mtp', 'all'],
                        help='Which model to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    args = parser.parse_args()

    create_dirs()

    if args.model == 'all':
        models_to_train = ['lstm_seq2seq', 'transformer', 'v2x_graph', 'co_mtp']
    else:
        models_to_train = [args.model]

    all_results = {}
    for model_name in models_to_train:
        results = train_model(model_name, args.epochs, args.batch_size)
        all_results[model_name] = results
        print("\n")

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {combined_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Cooperative':<12} {'Params':<12} "
          f"{'minADE':<10} {'minFDE':<10} {'MR':<10}")
    print("-" * 80)
    for name, res in all_results.items():
        tm = res['test_metrics']
        print(f"{res['model_display_name']:<20} "
              f"{'Yes' if res['cooperative'] else 'No':<12} "
              f"{res['num_parameters']:>10,} "
              f"{tm['minADE']:>9.4f} "
              f"{tm['minFDE']:>9.4f} "
              f"{tm['MR']:>9.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
