"""Evaluation utilities for LSTM frequency filter.

Computes MSE metrics and generates predictions for analysis.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, List

from ..models.lstm_filter import LSTMFrequencyFilter


class ModelEvaluator:
    """Evaluator for computing metrics and generating predictions."""
    
    def __init__(self, model: LSTMFrequencyFilter, device: torch.device):
        """Initialize evaluator.
        
        Args:
            model: Trained LSTM model
            device: Device (CPU/CUDA)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def compute_mse(self, data_loader: DataLoader) -> float:
        """Compute MSE on dataset.
        
        Formula: MSE = (1/N)Σ(LSTM(S[t], C) - Target[t])²
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            mse: Mean Squared Error
        """
        criterion = nn.MSELoss()
        total_loss = 0.0
        num_samples = 0
        
        batch_size = data_loader.batch_size
        hidden = self.model.init_hidden(batch_size, self.device)
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle variable batch size
                current_batch_size = inputs.size(0)
                if current_batch_size != batch_size:
                    hidden = self.model.init_hidden(current_batch_size, self.device)
                
                # Forward pass
                output, hidden = self.model(inputs, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                # Compute loss
                output = output.squeeze(1)
                loss = criterion(output, targets)
                
                total_loss += loss.item() * current_batch_size
                num_samples += current_batch_size
        
        mse = total_loss / num_samples
        return mse
    
    def generate_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions for entire dataset.
        
        Args:
            data_loader: DataLoader for prediction
        
        Returns:
            predictions: Model predictions [N]
            targets: Ground truth targets [N]
        """
        predictions_list = []
        targets_list = []
        
        batch_size = data_loader.batch_size
        hidden = self.model.init_hidden(batch_size, self.device)
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle variable batch size
                current_batch_size = inputs.size(0)
                if current_batch_size != batch_size:
                    hidden = self.model.init_hidden(current_batch_size, self.device)
                
                # Forward pass
                output, hidden = self.model(inputs, hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                # Store predictions and targets
                output = output.squeeze(1)
                predictions_list.append(output.cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        predictions = np.concatenate(predictions_list, axis=0).flatten()
        targets = np.concatenate(targets_list, axis=0).flatten()
        
        return predictions, targets
    
    def evaluate_per_frequency(
        self,
        data_loader: DataLoader,
        num_frequencies: int = 4
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-frequency metrics.
        
        Args:
            data_loader: DataLoader for evaluation
            num_frequencies: Number of frequencies (default: 4)
        
        Returns:
            metrics: Dictionary mapping frequency index to metrics dict
                    {freq_idx: {'mse': value, 'mae': value}}
        """
        # Generate all predictions
        predictions, targets = self.generate_predictions(data_loader)
        
        # Compute metrics per frequency
        # Each frequency has 10,000 samples
        samples_per_freq = len(predictions) // num_frequencies
        metrics = {}
        
        for freq_idx in range(num_frequencies):
            start_idx = freq_idx * samples_per_freq
            end_idx = start_idx + samples_per_freq
            
            freq_preds = predictions[start_idx:end_idx]
            freq_targets = targets[start_idx:end_idx]
            
            # Compute MSE and MAE
            mse = np.mean((freq_preds - freq_targets) ** 2)
            mae = np.mean(np.abs(freq_preds - freq_targets))
            
            metrics[freq_idx] = {
                'mse': float(mse),
                'mae': float(mae)
            }
        
        return metrics
    
    def evaluate_generalization(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate generalization by comparing train and test MSE.
        
        Good generalization: MSE_test ≈ MSE_train
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
        
        Returns:
            metrics: Dictionary with train_mse, test_mse, and generalization_gap
        """
        train_mse = self.compute_mse(train_loader)
        test_mse = self.compute_mse(test_loader)
        
        # Generalization gap: difference between test and train MSE
        gap = test_mse - train_mse
        
        metrics = {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'generalization_gap': float(gap),
            'generalizes_well': abs(gap) < 0.01  # Threshold for good generalization
        }
        
        return metrics
    
    def extract_frequency_predictions(
        self,
        data_loader: DataLoader,
        freq_idx: int,
        num_frequencies: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract predictions for a specific frequency.
        
        Useful for per-frequency visualization.
        
        Args:
            data_loader: DataLoader for evaluation
            freq_idx: Frequency index (0-3)
            num_frequencies: Total number of frequencies
        
        Returns:
            freq_predictions: Predictions for this frequency [10000]
            freq_targets: Targets for this frequency [10000]
        """
        predictions, targets = self.generate_predictions(data_loader)
        
        # Extract samples for specific frequency
        samples_per_freq = len(predictions) // num_frequencies
        start_idx = freq_idx * samples_per_freq
        end_idx = start_idx + samples_per_freq
        
        freq_predictions = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        return freq_predictions, freq_targets


def evaluate_model(
    model: LSTMFrequencyFilter,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Comprehensive model evaluation.
    
    Args:
        model: Trained LSTM model
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device (CPU/CUDA)
    
    Returns:
        results: Dictionary containing all evaluation metrics
    """
    evaluator = ModelEvaluator(model, device)
    
    # Overall metrics
    train_mse = evaluator.compute_mse(train_loader)
    test_mse = evaluator.compute_mse(test_loader)
    
    # Generalization metrics
    gen_metrics = evaluator.evaluate_generalization(train_loader, test_loader)
    
    # Per-frequency metrics
    train_freq_metrics = evaluator.evaluate_per_frequency(train_loader)
    test_freq_metrics = evaluator.evaluate_per_frequency(test_loader)
    
    results = {
        'overall': {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse)
        },
        'generalization': gen_metrics,
        'per_frequency': {
            'train': train_freq_metrics,
            'test': test_freq_metrics
        }
    }
    
    return results
