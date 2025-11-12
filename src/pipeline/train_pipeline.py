"""Training pipeline execution."""

import torch
from pathlib import Path
from typing import Dict, Tuple

from ..data.generator import SignalGenerator, create_train_test_datasets
from ..data.dataset import create_dataloaders, save_dataset
from ..models.model_factory import create_model
from ..training.trainer import LSTMTrainer
from ..training.evaluator import ModelEvaluator


def execute_training_pipeline(
    train_seed: int,
    test_seed: int,
    frequencies: list,
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    device: torch.device,
    save_dir: Path,
    verbose: bool = True
) -> Dict:
    """Execute complete training pipeline.
    
    Args:
        train_seed: Random seed for training data
        test_seed: Random seed for test data
        frequencies: List of frequencies to extract
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
        device: Device to train on
        save_dir: Directory to save outputs
        verbose: Print progress
        
    Returns:
        Dictionary with training results
    """
    if verbose:
        print("=" * 80)
        print("Training Pipeline Execution")
        print("=" * 80)
    
    train_data, test_data = create_train_test_datasets(train_seed, test_seed)
    S_train, targets_train, one_hot_train = train_data
    S_test, targets_test, one_hot_test = test_data
    
    if verbose:
        print(f"[INFO] Training samples: {len(targets_train)}")
        print(f"[INFO] Test samples: {len(targets_test)}")
    
    dataset_dir = save_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(S_train, targets_train, one_hot_train,  dataset_dir / f"train_data_seed{train_seed}.pt")
    save_dataset(S_test, targets_test, one_hot_test, dataset_dir / f"test_data_seed{test_seed}.pt")
    
    train_loader, test_loader = create_dataloaders(
        S_train, targets_train, one_hot_train,
        S_test, targets_test, one_hot_test,
        batch_size=batch_size,
        shuffle_train=False
    )
    
    model = create_model(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=0.2,
        device=device
    )
    
    if verbose:
        print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = LSTMTrainer(model, device, learning_rate)
    
    model_dir = save_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    history = trainer.train(
        train_loader, test_loader,
        num_epochs=num_epochs,
        save_path=str(model_dir),
        save_every=20,
        verbose=verbose
    )
    
    final_path = model_dir / f"lstm_l1_epoch{num_epochs}_final.pth"
    trainer._save_checkpoint(final_path, num_epochs)
    
    evaluator = ModelEvaluator(model, device)
    train_mse = evaluator.compute_mse(train_loader)
    test_mse = evaluator.compute_mse(test_loader)
    train_freq_metrics = evaluator.evaluate_per_frequency(train_loader)
    test_freq_metrics = evaluator.evaluate_per_frequency(test_loader)
    
    train_predictions, train_targets = evaluator.generate_predictions(train_loader)
    test_predictions, test_targets = evaluator.generate_predictions(test_loader)
    
    if verbose:
        print(f"\n[RESULTS] Train MSE: {train_mse:.8f}")
        print(f"[RESULTS] Test MSE: {test_mse:.8f}")
        print(f"[RESULTS] Generalization Gap: {abs(test_mse - train_mse):.8f}")
    
    return {
        'history': history,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_freq_metrics': train_freq_metrics,
        'test_freq_metrics': test_freq_metrics,
        'train_predictions': train_predictions,
        'train_targets': train_targets,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'train_data': train_data,
        'test_data': test_data
    }

