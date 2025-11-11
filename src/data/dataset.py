"""PyTorch Dataset for LSTM frequency extraction.

Implements dataset structure with:
- Input: 5-dimensional vector [S(t), C1, C2, C3, C4]
- Output: Scalar pure frequency value
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


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
        """Initialize dataset.
        
        Args:
            S: Noisy mixed signal expanded to [40000] (each time sample repeated 4 times)
            targets: Pure frequency targets [40000] (flattened from [4, 10000])
            one_hot_vectors: Frequency selection vectors [40000, 4]
        """
        # Convert to torch tensors
        self.S = torch.tensor(S, dtype=torch.float32).unsqueeze(1)  # [40000, 1]
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)  # [40000, 1]
        self.one_hot = torch.tensor(one_hot_vectors, dtype=torch.float32)  # [40000, 4]
        
        # Concatenate to create input vector [S(t), C1, C2, C3, C4]
        self.inputs = torch.cat([self.S, self.one_hot], dim=1)  # [40000, 5]
        
        # For LSTM with sequence_length=1, add sequence dimension
        self.inputs = self.inputs.unsqueeze(1)  # [40000, 1, 5] for (batch, seq_len, features)
        
    def __len__(self) -> int:
        """Return total number of samples (40,000)."""
        return len(self.inputs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single training sample.
        
        Args:
            idx: Sample index
        
        Returns:
            input: [1, 5] tensor containing [S(t), C1, C2, C3, C4]
            target: [1] scalar target value
        """
        return self.inputs[idx], self.targets[idx]
    
    def get_by_frequency(self, freq_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract all samples for a specific frequency.
        
        Useful for per-frequency analysis and visualization.
        
        Args:
            freq_idx: Frequency index (0-3)
        
        Returns:
            inputs: All inputs for this frequency [10000, 1, 5]
            targets: All targets for this frequency [10000, 1]
        """
        # Get indices where one-hot vector has 1 at freq_idx
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
    shuffle_train: bool = False,  # IMPORTANT: Usually False for sequential L=1 training
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create training and test DataLoaders.
    
    CRITICAL NOTE: For L=1 with state management, shuffle should be False
    to maintain temporal continuity. If shuffle=True, hidden states become
    less meaningful as samples are not consecutive.
    
    Args:
        S_train: Training noisy signal [40000]
        targets_train: Training targets [40000]
        one_hot_train: Training one-hot vectors [40000, 4]
        S_test: Test noisy signal [40000]
        targets_test: Test targets [40000]
        one_hot_test: Test one-hot vectors [40000, 4]
        batch_size: Batch size for training
        shuffle_train: Whether to shuffle training data (False for sequential L=1)
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
    """
    # Create datasets
    train_dataset = FrequencyDataset(S_train, targets_train, one_hot_train)
    test_dataset = FrequencyDataset(S_test, targets_test, one_hot_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Never shuffle test data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
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
    data = torch.load(filepath)
    return data['S'], data['targets'], data['one_hot']
