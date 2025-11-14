"""Data transformation utilities."""

import torch
import numpy as np
from typing import Tuple


def prepare_dataset_tensors(
    S: np.ndarray,
    targets: np.ndarray,
    one_hot_vectors: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert numpy arrays to torch tensors with proper shapes.
    
    Args:
        S: Noisy mixed signal [40000]
        targets: Pure frequency targets [40000]
        one_hot_vectors: Frequency selection vectors [40000, 4]
    
    Returns:
        S_tensor: Signal tensor [40000, 1]
        targets_tensor: Targets tensor [40000, 1]
        one_hot_tensor: One-hot tensor [40000, 4]
    """
    S_tensor = torch.tensor(S, dtype=torch.float32).unsqueeze(1)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    one_hot_tensor = torch.tensor(one_hot_vectors, dtype=torch.float32)
    return S_tensor, targets_tensor, one_hot_tensor


def create_lstm_input(
    S_tensor: torch.Tensor,
    one_hot_tensor: torch.Tensor
) -> torch.Tensor:
    """Create LSTM input by concatenating signal and one-hot vectors.
    
    Args:
        S_tensor: Signal tensor [40000, 1]
        one_hot_tensor: One-hot tensor [40000, 4]
    
    Returns:
        inputs: LSTM inputs [40000, 1, 5] for (batch, seq_len, features)
    """
    inputs = torch.cat([S_tensor, one_hot_tensor], dim=1)
    inputs = inputs.unsqueeze(1)
    return inputs


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
):
    """Create DataLoader with consistent parameters.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

