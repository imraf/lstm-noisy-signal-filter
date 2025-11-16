"""PyTorch Dataset for LSTM frequency extraction."""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

from .data_utils import prepare_dataset_tensors, create_lstm_input, create_dataloader


class FrequencyDataset(Dataset):
    """PyTorch Dataset for conditional frequency extraction.
    
    Each sample consists of:
    - Input: [S(t), C1, C2, C3, C4] where S(t) is noisy signal sample 
             and C is one-hot selection vector
    - Target: Pure sine wave value at time t for selected frequency
    
    Total samples: 40,000 (10,000 time steps × 4 frequencies)
    """
    
    def __init__(
        self,
        S: np.ndarray,
        targets: np.ndarray,
        one_hot_vectors: np.ndarray
    ):
        """Initialize dataset."""
        self.S, self.targets, self.one_hot = prepare_dataset_tensors(S, targets, one_hot_vectors)
        self.inputs = create_lstm_input(self.S, self.one_hot)
        
    def __len__(self) -> int:
        """Return total number of samples (40,000)."""
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single training sample."""
        return self.inputs[idx], self.targets[idx]
    
    def get_by_frequency(self, freq_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract all samples for a specific frequency."""
        indices = torch.where(self.one_hot[:, freq_idx] == 1.0)[0]
        return self.inputs[indices], self.targets[indices]


def create_dataloaders(
    S_train: np.ndarray,
    targets_train: np.ndarray,
    one_hot_train: np.ndarray,
    S_test: np.ndarray,
    targets_test: np.ndarray,
    one_hot_test: np.ndarray,
    batch_size: int = 32,
    shuffle_train: bool = False,
    num_workers: int = 0
) -> Tuple:
    """Create training and test DataLoaders."""
    train_dataset = FrequencyDataset(S_train, targets_train, one_hot_train)
    test_dataset = FrequencyDataset(S_test, targets_test, one_hot_test)
    train_loader = create_dataloader(train_dataset, batch_size, shuffle_train, num_workers)
    test_loader = create_dataloader(test_dataset, batch_size, False, num_workers)
    return train_loader, test_loader


def save_dataset(
    S: np.ndarray,
    targets: np.ndarray,
    one_hot: np.ndarray,
    filepath: str
):
    """Save dataset to file.
    
    Args:
        S: Noisy signal [40000]
        targets: Pure targets [40000]
        one_hot: One-hot vectors [40000, 4]
        filepath: Path to save file (e.g., 'outputs/datasets/train_data_seed11.pt')
    """
    torch.save({
        'S': S,
        'targets': targets,
        'one_hot': one_hot
    }, filepath)


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset from file.
    
    Args:
        filepath: Path to saved dataset
    
    Returns:
        S: Noisy signal [40000]
        targets: Pure targets [40000]
        one_hot: One-hot vectors [40000, 4]
    """
    data = torch.load(filepath, weights_only=False)
    return data['S'], data['targets'], data['one_hot']
