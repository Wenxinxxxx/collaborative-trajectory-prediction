"""
Loss functions for multi-modal trajectory prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WinnerTakeAllLoss(nn.Module):
    """
    Winner-Take-All (WTA) loss for multi-modal trajectory prediction.

    For each sample, only the best mode (closest to GT) is penalized.
    This encourages diverse multi-modal predictions.

    L = min_k ||pred_k - gt||^2 + lambda * CE(mode_prob, best_k)
    """

    def __init__(self, lambda_cls=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls

    def forward(self, predictions, mode_probs, gt):
        """
        Args:
            predictions: (B, K, T_f, 2) multi-modal predictions
            mode_probs:  (B, K) predicted mode probabilities
            gt:          (B, T_f, 2) ground truth

        Returns:
            total_loss: scalar
            reg_loss:   regression loss (scalar)
            cls_loss:   classification loss (scalar)
        """
        B, K, T, D = predictions.shape

        # Compute ADE for each mode
        gt_expanded = gt.unsqueeze(1).expand_as(predictions)  # (B, K, T, 2)
        errors = torch.sqrt(
            torch.sum((predictions - gt_expanded) ** 2, dim=-1) + 1e-8
        )  # (B, K, T)
        ade_per_mode = errors.mean(dim=-1)  # (B, K)

        # Find best mode for each sample
        best_mode = ade_per_mode.argmin(dim=-1)  # (B,)

        # Regression loss: only on best mode
        best_pred = predictions[torch.arange(B), best_mode]  # (B, T, 2)
        reg_loss = F.smooth_l1_loss(best_pred, gt)

        # Classification loss: encourage high probability for best mode
        # Use log_softmax for numerical stability
        log_probs = torch.log(mode_probs.clamp(min=1e-6))
        cls_loss = F.nll_loss(log_probs, best_mode)

        total_loss = reg_loss + self.lambda_cls * cls_loss

        return total_loss, reg_loss, cls_loss


class MultiModalLoss(nn.Module):
    """
    Combined loss with Gaussian NLL for regression and CE for classification.
    """

    def __init__(self, lambda_cls=1.0):
        super().__init__()
        self.lambda_cls = lambda_cls

    def forward(self, predictions, mode_probs, gt):
        """Same interface as WinnerTakeAllLoss."""
        B, K, T, D = predictions.shape

        gt_expanded = gt.unsqueeze(1).expand_as(predictions)
        errors = torch.sum((predictions - gt_expanded) ** 2, dim=-1)
        ade_per_mode = errors.mean(dim=-1)

        best_mode = ade_per_mode.argmin(dim=-1)

        # Weighted regression loss
        weights = F.softmax(-ade_per_mode.detach(), dim=-1)  # (B, K)
        reg_loss = (weights * ade_per_mode).sum(dim=-1).mean()

        # Classification loss
        cls_loss = F.cross_entropy(
            torch.log(mode_probs + 1e-8), best_mode
        )

        total_loss = reg_loss + self.lambda_cls * cls_loss
        return total_loss, reg_loss, cls_loss
