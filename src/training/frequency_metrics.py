"""Per-frequency metrics computation."""

import numpy as np
from typing import Dict


def compute_per_frequency_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_frequencies: int = 4
) -> Dict[int, Dict[str, float]]:
    """Compute per-frequency metrics.
    
    Args:
        predictions: All predictions
        targets: All targets
        num_frequencies: Number of frequencies
    
    Returns:
        metrics: Dictionary mapping frequency index to metrics
    """
    samples_per_freq = len(predictions) // num_frequencies
    metrics = {}
    
    for freq_idx in range(num_frequencies):
        start_idx = freq_idx * samples_per_freq
        end_idx = start_idx + samples_per_freq
        
        freq_preds = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        mse = np.mean((freq_preds - freq_targets) ** 2)
        mae = np.mean(np.abs(freq_preds - freq_targets))
        
        metrics[freq_idx] = {
            'mse': float(mse),
            'mae': float(mae)
        }
    
    return metrics


def extract_frequency_samples(
    predictions: np.ndarray,
    targets: np.ndarray,
    freq_idx: int,
    num_frequencies: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Extract samples for specific frequency.
    
    Args:
        predictions: All predictions
        targets: All targets
        freq_idx: Frequency index (0-3)
        num_frequencies: Total number of frequencies
    
    Returns:
        freq_predictions: Predictions for this frequency
        freq_targets: Targets for this frequency
    """
    samples_per_freq = len(predictions) // num_frequencies
    start_idx = freq_idx * samples_per_freq
    end_idx = start_idx + samples_per_freq
    
    return predictions[start_idx:end_idx], targets[start_idx:end_idx]

