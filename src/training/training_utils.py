"""Utility functions for LSTM training."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def process_batch(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    hidden: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    is_training: bool = True
) -> Tuple[float, Tuple[torch.Tensor, torch.Tensor]]:
    """Process a single batch.
    
    Args:
        model: LSTM model
        inputs: Input tensor
        targets: Target tensor
        hidden: Hidden state tuple
        criterion: Loss function
        device: Device
        is_training: Whether in training mode
    
    Returns:
        loss_value: Loss value for batch
        hidden: Updated hidden state
    """
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    output, hidden = model(inputs, hidden)
    hidden = (hidden[0].detach(), hidden[1].detach())
    
    output = output.squeeze(1)
    loss = criterion(output, targets)
    
    return loss.item(), hidden


def get_or_reset_hidden(
    model: nn.Module,
    current_batch_size: int,
    expected_batch_size: int,
    hidden: Tuple[torch.Tensor, torch.Tensor],
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get hidden state or reset if batch size changed.
    
    Args:
        model: LSTM model
        current_batch_size: Current batch size
        expected_batch_size: Expected batch size
        hidden: Current hidden state
        device: Device
    
    Returns:
        hidden: Hidden state (new if size changed)
    """
    if current_batch_size != expected_batch_size:
        return model.init_hidden(current_batch_size, device)
    return hidden

