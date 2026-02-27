"""
Evaluation metrics for trajectory prediction.
Implements minADE, minFDE, and Miss Rate (MR).
"""

import torch
import numpy as np


def compute_ade(pred, gt):
    """
    Compute Average Displacement Error for a single mode.

    Args:
        pred: (future_steps, 2) predicted trajectory
        gt:   (future_steps, 2) ground truth trajectory

    Returns:
        float: ADE value in meters
    """
    errors = torch.sqrt(torch.sum((pred - gt) ** 2, dim=-1))
    return torch.mean(errors).item()


def compute_fde(pred, gt):
    """
    Compute Final Displacement Error for a single mode.

    Args:
        pred: (future_steps, 2) predicted trajectory
        gt:   (future_steps, 2) ground truth trajectory

    Returns:
        float: FDE value in meters
    """
    error = torch.sqrt(torch.sum((pred[-1] - gt[-1]) ** 2, dim=-1))
    return error.item()


def compute_min_ade(preds, gt):
    """
    Compute minimum ADE across multiple prediction modes.

    Args:
        preds: (num_modes, future_steps, 2) multi-modal predictions
        gt:    (future_steps, 2) ground truth trajectory

    Returns:
        float: minADE value in meters
    """
    if preds.dim() == 2:
        return compute_ade(preds, gt)

    num_modes = preds.shape[0]
    ades = []
    for k in range(num_modes):
        ades.append(compute_ade(preds[k], gt))
    return min(ades)


def compute_min_fde(preds, gt):
    """
    Compute minimum FDE across multiple prediction modes.

    Args:
        preds: (num_modes, future_steps, 2) multi-modal predictions
        gt:    (future_steps, 2) ground truth trajectory

    Returns:
        float: minFDE value in meters
    """
    if preds.dim() == 2:
        return compute_fde(preds, gt)

    num_modes = preds.shape[0]
    fdes = []
    for k in range(num_modes):
        fdes.append(compute_fde(preds[k], gt))
    return min(fdes)


def compute_miss_rate(preds, gt, threshold=2.0):
    """
    Compute Miss Rate: whether the best FDE exceeds a threshold.

    Args:
        preds: (num_modes, future_steps, 2) multi-modal predictions
        gt:    (future_steps, 2) ground truth trajectory
        threshold: distance threshold in meters

    Returns:
        float: 1.0 if miss, 0.0 if hit
    """
    min_fde = compute_min_fde(preds, gt)
    return 1.0 if min_fde > threshold else 0.0


def compute_batch_metrics(preds_batch, gt_batch, num_modes=1):
    """
    Compute all metrics for a batch of predictions.

    Args:
        preds_batch: (batch, num_modes, future_steps, 2) or (batch, future_steps, 2)
        gt_batch:    (batch, future_steps, 2)
        num_modes:   number of prediction modes

    Returns:
        dict: {'minADE': float, 'minFDE': float, 'MR': float}
    """
    batch_size = gt_batch.shape[0]
    total_ade = 0.0
    total_fde = 0.0
    total_mr = 0.0

    for i in range(batch_size):
        if preds_batch.dim() == 4:
            pred = preds_batch[i]  # (num_modes, future_steps, 2)
        else:
            pred = preds_batch[i].unsqueeze(0)  # (1, future_steps, 2)
        gt = gt_batch[i]  # (future_steps, 2)

        total_ade += compute_min_ade(pred, gt)
        total_fde += compute_min_fde(pred, gt)
        total_mr += compute_miss_rate(pred, gt)

    return {
        'minADE': total_ade / batch_size,
        'minFDE': total_fde / batch_size,
        'MR': total_mr / batch_size,
    }
