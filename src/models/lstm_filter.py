"""LSTM model for frequency extraction from mixed noisy signals.

Implements LSTM architecture with:
- Input: 5-dimensional vector [S(t), C1, C2, C3, C4]
- Output: Single scalar (pure frequency value)
- Critical state management for L=1 training
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LSTMFrequencyFilter(nn.Module):
    """LSTM network for extracting pure frequencies from mixed noisy signals.
    
    Architecture:
    - Input layer: 5 features [S(t), C1, C2, C3, C4]
    - LSTM layers: Configurable hidden size and number of layers
    - Output layer: Single scalar prediction
    
    CRITICAL: For L=1 training, hidden state (h_t, c_t) must be manually
    managed between consecutive samples to enable temporal learning.
    """
    
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        """Initialize LSTM frequency filter.
        
        Args:
            input_size: Dimension of input vector (default: 5)
            hidden_size: Number of features in hidden state
            num_layers: Number of stacked LSTM layers
            output_size: Dimension of output (default: 1)
            dropout: Dropout probability between LSTM layers
        """
        super(LSTMFrequencyFilter, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input shape: (batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through LSTM.
        
        Args:
            x: Input tensor of shape [batch, seq_len, input_size]
               For L=1: [batch, 1, 5]
            hidden: Optional tuple (h_0, c_0) of hidden and cell states
                   Each of shape [num_layers, batch, hidden_size]
                   If None, initialized to zeros
        
        Returns:
            output: Predictions of shape [batch, seq_len, output_size]
                   For L=1: [batch, 1, 1]
            (h_n, c_n): Updated hidden and cell states
        """
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Apply fully connected layer to get predictions
        # lstm_out shape: [batch, seq_len, hidden_size]
        output = self.fc(lstm_out)
        # output shape: [batch, seq_len, output_size]
        
        return output, (h_n, c_n)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states with zeros.
        
        Args:
            batch_size: Batch size
            device: Device to create tensors on (CPU or CUDA)
        
        Returns:
            (h_0, c_0): Tuple of initial hidden and cell states
                       Each of shape [num_layers, batch_size, hidden_size]
        """
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h_0, c_0
    
    def predict(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions without computing gradients.
        
        Useful for evaluation and inference.
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            hidden: Optional hidden state tuple
        
        Returns:
            output: Predictions [batch, seq_len, output_size]
            hidden: Updated hidden state tuple
        """
        self.eval()
        with torch.no_grad():
            output, hidden = self.forward(x, hidden)
        return output, hidden


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
    
    # Create model with saved configuration
    config = checkpoint['model_config']
    model = LSTMFrequencyFilter(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=config['output_size']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    epoch = checkpoint.get('epoch', 0)
    
    return model, epoch
