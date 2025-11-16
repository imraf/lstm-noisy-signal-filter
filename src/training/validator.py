"""Validation logic for LSTM training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List

from ..models.lstm_filter import LSTMFrequencyFilter
from .training_utils import get_or_reset_hidden


class Validator:
    """Handles model validation."""
    
    def __init__(
        self,
        model: LSTMFrequencyFilter,
        device: torch.device,
        criterion: nn.Module
    ):
        """Initialize validator.
        
        Args:
            model: LSTM model
            device: Device (CPU/CUDA)
            criterion: Loss function
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.val_losses: List[float] = []
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model on validation/test set.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        batch_size = val_loader.batch_size
        hidden = self.model.init_hidden(batch_size, self.device)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                current_batch_size = inputs.size(0)
                hidden = get_or_reset_hidden(
                    self.model, current_batch_size, batch_size, hidden, self.device
                )
                
                output, hidden = self.model(inputs.to(self.device), hidden)
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                output = output.squeeze(1)
                loss = self.criterion(output, targets.to(self.device))
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss

