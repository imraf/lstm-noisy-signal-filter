"""Factory functions for LSTM model creation, saving, and loading."""

import torch
from typing import Optional, Tuple
from .lstm_filter import LSTMFrequencyFilter


def create_model(
    input_size: int = 5,
    hidden_size: int = 64,
    num_layers: int = 2,
    output_size: int = 1,
    dropout: float = 0.2,
    device: Optional[torch.device] = None
) -> LSTMFrequencyFilter:
    """Factory function to create and initialize LSTM model.
    
    Args:
        input_size: Input dimension (default: 5)
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        output_size: Output dimension (default: 1)
        dropout: Dropout rate
        device: Target device (CPU/CUDA)
    
    Returns:
        model: Initialized LSTM model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMFrequencyFilter(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    )
    
    model = model.to(device)
    return model


def save_model(model: LSTMFrequencyFilter, filepath: str, epoch: int, optimizer_state: dict = None):
    """Save model checkpoint.
    
    Args:
        model: LSTM model to save
        filepath: Path to save checkpoint
        epoch: Current epoch number
        optimizer_state: Optional optimizer state dict
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'output_size': model.output_size
        }
    }
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    torch.save(checkpoint, filepath)


def load_model(filepath: str, device: Optional[torch.device] = None) -> Tuple[LSTMFrequencyFilter, int]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Target device
    
    Returns:
        model: Loaded LSTM model
        epoch: Epoch number from checkpoint
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    config = checkpoint['model_config']
    model = LSTMFrequencyFilter(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    epoch = checkpoint.get('epoch', 0)
    
    return model, epoch

