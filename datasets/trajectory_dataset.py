"""
PyTorch Dataset and DataLoader for trajectory prediction.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


class TrajectoryDataset(Dataset):
    """
    Dataset for trajectory prediction.

    Each sample contains:
      - history:       (history_steps, 2)  ego agent's past trajectory
      - future:        (future_steps, 2)   ego agent's future trajectory (GT)
      - neighbors:     (max_agents, history_steps, 2) neighbor trajectories
      - neighbor_mask: (max_agents,)        which neighbors are valid
      - infra_history: (history_steps, 2)  infrastructure view of ego
      - infra_mask:    scalar              whether infra view is available
      - lanes:         (max_lanes, lane_points, 2) lane centerlines
      - lane_mask:     (max_lanes,)         which lanes are valid
      - origin:        (2,)                 origin for de-normalization
    """

    def __init__(self, data_path):
        """
        Args:
            data_path: path to .pt file containing list of sample dicts
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
        self.samples = torch.load(data_path, weights_only=False)
        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """
    Custom collate function to handle dictionary samples.

    Returns a dictionary of batched tensors.
    """
    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([sample[key] for sample in batch])
        elif isinstance(batch[0][key], (str, int, float)):
            collated[key] = [sample[key] for sample in batch]
        else:
            collated[key] = [sample[key] for sample in batch]

    return collated


def get_dataloaders(batch_size=None, num_workers=0):
    """
    Create train, validation, and test DataLoaders.

    Args:
        batch_size: override default batch size
        num_workers: number of data loading workers

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    train_path = os.path.join(PROCESSED_DATA_DIR, 'train_samples.pt')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'val_samples.pt')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'test_samples.pt')

    train_dataset = TrajectoryDataset(train_path)
    val_dataset = TrajectoryDataset(val_path)
    test_dataset = TrajectoryDataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)

    for batch in train_loader:
        print("\nSample batch contents:")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
            else:
                print(f"  {key}: type={type(val[0])}, len={len(val)}")
        break
