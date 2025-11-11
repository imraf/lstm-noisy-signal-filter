"""Training loop with critical LSTM state management for L=1.

CRITICAL: When L=1, hidden state (h_t, c_t) must be preserved between
consecutive samples and detached to prevent gradient explosion.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from pathlib import Path

from ..models.lstm_filter import LSTMFrequencyFilter
from .checkpoint_manager import CheckpointManager
from .training_utils import get_or_reset_hidden
from .validator import Validator


class LSTMTrainer:
    """Trainer for LSTM frequency filter with proper state management."""
    
    def __init__(
        self,
        model: LSTMFrequencyFilter,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0
    ):
        """Initialize trainer."""
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.train_losses: List[float] = []
        self.checkpoint_manager = CheckpointManager()
        self.validator = Validator(model, device, self.criterion)
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        reset_state_each_batch: bool = False
    ) -> float:
        """Train for one epoch. Hidden state preserved across batches unless reset_state_each_batch=True."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        batch_size = train_loader.batch_size
        hidden = self.model.init_hidden(batch_size, self.device)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            current_batch_size = inputs.size(0)
            hidden = get_or_reset_hidden(
                self.model, current_batch_size, batch_size, hidden, self.device
            )
            
            if reset_state_each_batch:
                hidden = self.model.init_hidden(current_batch_size, self.device)
            
            self.optimizer.zero_grad()
            
            output, hidden = self.model(inputs.to(self.device), hidden)
            hidden = (hidden[0].detach(), hidden[1].detach())
            
            output = output.squeeze(1)
            loss = self.criterion(output, targets.to(self.device))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
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
        """Full training loop. Returns history with 'train_loss' and 'val_loss' lists."""
        if save_path:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validator.validate(val_loader) if val_loader else None
            
            if verbose:
                msg = f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
            
            if save_path and (epoch % save_every == 0 or epoch == num_epochs):
                self.checkpoint_manager.save_dir = save_dir
                checkpoint_path = self.checkpoint_manager.get_checkpoint_path(epoch)
                self._save_checkpoint(checkpoint_path, epoch)
                if verbose:
                    print(f"  Checkpoint saved: {checkpoint_path}")
        
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.validator.val_losses if self.validator.val_losses else None
        }
        
        return history
    
    def _save_checkpoint(self, filepath: str, epoch: int):
        """Save training checkpoint."""
        model_config = {
            'input_size': self.model.input_size,
            'hidden_size': self.model.hidden_size,
            'num_layers': self.model.num_layers,
            'output_size': self.model.output_size
        }
        self.checkpoint_manager.save_checkpoint(
            filepath, epoch,
            self.model.state_dict(),
            self.optimizer.state_dict(),
            self.train_losses,
            self.validator.val_losses,
            model_config
        )
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = self.checkpoint_manager.load_checkpoint(filepath, self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.validator.val_losses = checkpoint.get('val_losses', [])
        return checkpoint.get('epoch', 0)
