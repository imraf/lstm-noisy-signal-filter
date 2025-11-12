"""Prediction generation for LSTM model."""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple

from ..models.lstm_filter import LSTMFrequencyFilter


def generate_predictions(
    model: LSTMFrequencyFilter,
    data_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions for entire dataset.
    
    Args:
        model: LSTM model
        data_loader: DataLoader for prediction
        device: Device
    
    Returns:
        predictions: Model predictions [N]
        targets: Ground truth targets [N]
    """
    model.eval()
    predictions_list = []
    targets_list = []
    
    batch_size = data_loader.batch_size
    hidden = model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            current_batch_size = inputs.size(0)
            if current_batch_size != batch_size:
                hidden = model.init_hidden(current_batch_size, device)
            
            output, hidden = model(inputs, hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            output = output.squeeze(1)
            # Use numpy(force=True) to handle PyTorch 2.2.x compatibility
            try:
                predictions_list.append(output.detach().cpu().numpy(force=True))
                targets_list.append(targets.detach().cpu().numpy(force=True))
            except (RuntimeError, TypeError):
                # Fallback for older PyTorch versions
                predictions_list.append(np.array(output.detach().cpu().tolist()))
                targets_list.append(np.array(targets.detach().cpu().tolist()))
    
    predictions = np.concatenate(predictions_list, axis=0).flatten()
    targets = np.concatenate(targets_list, axis=0).flatten()
    
    return predictions, targets

