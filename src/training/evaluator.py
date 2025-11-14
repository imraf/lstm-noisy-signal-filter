"""Evaluation utilities for LSTM frequency filter."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple

from ..models.lstm_filter import LSTMFrequencyFilter
from .prediction_generator import generate_predictions
from .frequency_metrics import compute_per_frequency_metrics, extract_frequency_samples


class ModelEvaluator:
    """Evaluator for computing metrics and generating predictions."""
    
    def __init__(self, model: LSTMFrequencyFilter, device: torch.device):
        """Initialize evaluator."""
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def compute_mse(self, data_loader: DataLoader) -> float:
        """Compute MSE on dataset."""
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
        """Generate predictions for entire dataset."""
        return generate_predictions(self.model, data_loader, self.device)
    
    def evaluate_per_frequency(
        self,
        data_loader: DataLoader,
        num_frequencies: int = 4
    ) -> Dict[int, Dict[str, float]]:
        """Compute per-frequency metrics."""
        predictions, targets = self.generate_predictions(data_loader)
        return compute_per_frequency_metrics(predictions, targets, num_frequencies)
    
    def evaluate_generalization(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate generalization by comparing train and test MSE."""
        train_mse = self.compute_mse(train_loader)
        test_mse = self.compute_mse(test_loader)
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
        """Extract predictions for a specific frequency."""
        predictions, targets = self.generate_predictions(data_loader)
        return extract_frequency_samples(predictions, targets, freq_idx, num_frequencies)


def evaluate_model(
    model: LSTMFrequencyFilter,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Comprehensive model evaluation."""
    evaluator = ModelEvaluator(model, device)
    train_mse = evaluator.compute_mse(train_loader)
    test_mse = evaluator.compute_mse(test_loader)
    gen_metrics = evaluator.evaluate_generalization(train_loader, test_loader)
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
