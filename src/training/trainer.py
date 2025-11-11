"""Training loop with critical LSTM state management for L=1.

CRITICAL: When L=1, hidden state (h_t, c_t) must be preserved between
consecutive samples and detached to prevent gradient explosion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

from ..models.lstm_filter import LSTMFrequencyFilter


class LSTMTrainer:
    """Trainer for LSTM frequency filter with proper state management.
    
    CRITICAL IMPLEMENTATION DETAIL:
    For L=1 training, we must:
    1. Initialize hidden state (h_0, c_0) at start of epoch
    2. Pass hidden state between consecutive batches
    3. Detach hidden state after each batch to prevent backprop through entire history
    4. DO NOT reset state between samples within an epoch
    """
    
    def __init__(
        self,
        model: LSTMFrequencyFilter,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0
    ):
        """Initialize trainer.
        
        Args:
            model: LSTM model to train
            device: Device (CPU/CUDA)
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization coefficient
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # MSE loss for regression
        self.criterion = nn.MSELoss()
        
        # Adam optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        reset_state_each_batch: bool = False
    ) -> float:
        """Train for one epoch with state management.
        
        CRITICAL: For L=1, hidden state should be preserved across batches
        unless reset_state_each_batch=True.
        
        Args:
            train_loader: Training data loader
            reset_state_each_batch: If True, reset hidden state for each batch.
                                   If False (default), preserve state across batches.
        
        Returns:
            avg_loss: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Initialize hidden state at start of epoch
        batch_size = train_loader.batch_size
        hidden = self.model.init_hidden(batch_size, self.device)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            inputs = inputs.to(self.device)  # [batch, seq_len=1, input_size=5]
            targets = targets.to(self.device)  # [batch, 1]
            
            # Handle variable batch size for last batch
            current_batch_size = inputs.size(0)
            if current_batch_size != batch_size:
                hidden = self.model.init_hidden(current_batch_size, self.device)
            
            # Reset state if requested (not typical for L=1 sequential training)
            if reset_state_each_batch:
                hidden = self.model.init_hidden(current_batch_size, self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output, hidden = self.model(inputs, hidden)
            
            # CRITICAL: Detach hidden state to prevent backprop through entire history
            # This prevents gradient explosion while preserving state values
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            # Compute loss
            # output shape: [batch, 1, 1], targets shape: [batch, 1]
            output = output.squeeze(1)  # [batch, 1]
            loss = self.criterion(output, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> float:
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
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Handle variable batch size
                current_batch_size = inputs.size(0)
                if current_batch_size != batch_size:
                    hidden = self.model.init_hidden(current_batch_size, self.device)
                
                # Forward pass
                output, hidden = self.model(inputs, hidden)
                
                # Detach state
                hidden = (hidden[0].detach(), hidden[1].detach())
                
                # Compute loss
                output = output.squeeze(1)
                loss = self.criterion(output, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        save_path: Optional[str] = None,
        save_every: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs to train
            save_path: Directory to save checkpoints
            save_every: Save checkpoint every N epochs
            verbose: Print training progress
        
        Returns:
            history: Dictionary with 'train_loss' and 'val_loss' lists
        """
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = None
            if val_loader:
                val_loss = self.validate(val_loader)
            
            # Print progress
            if verbose:
                msg = f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
            
            # Save checkpoint
            if save_path and (epoch % save_every == 0 or epoch == num_epochs):
                checkpoint_path = save_dir / f"lstm_l1_epoch{epoch}.pth"
                self.save_checkpoint(checkpoint_path, epoch)
                if verbose:
                    print(f"  Checkpoint saved: {checkpoint_path}")
        
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses if self.val_losses else None
        }
        
        return history
    
    def save_checkpoint(self, filepath: str, epoch: int):
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'input_size': self.model.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': self.model.output_size
            }
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        return checkpoint.get('epoch', 0)
