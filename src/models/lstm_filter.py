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
