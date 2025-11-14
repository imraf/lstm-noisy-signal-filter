"""Checkpoint management for LSTM training."""

import torch
from pathlib import Path
from typing import Dict, List, Optional


class CheckpointManager:
    """Manages model checkpoint saving and loading."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        filepath: str,
        epoch: int,
        model_state: Dict,
        optimizer_state: Dict,
        train_losses: List[float],
        val_losses: List[float],
        model_config: Dict
    ):
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            epoch: Current epoch number
            model_state: Model state dict
            optimizer_state: Optimizer state dict
            train_losses: Training loss history
            val_losses: Validation loss history
            model_config: Model configuration
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'model_config': model_config
        }
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str, device: torch.device) -> Dict:
        """Load training checkpoint.
        
        Args:
            filepath: Path to checkpoint file
            device: Device to load checkpoint to
        
        Returns:
            checkpoint: Dictionary containing checkpoint data
        """
        checkpoint = torch.load(filepath, map_location=device)
        return checkpoint
    
    def get_checkpoint_path(self, epoch: int, prefix: str = "lstm_l1") -> Path:
        """Generate checkpoint filepath.
        
        Args:
            epoch: Epoch number
            prefix: Filename prefix
        
        Returns:
            Path to checkpoint file
        """
        if self.save_dir is None:
            raise ValueError("save_dir not set")
        return self.save_dir / f"{prefix}_epoch{epoch}.pth"

