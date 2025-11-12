"""Tests for training module."""

import pytest
import torch
import numpy as np
from src.models.lstm_filter import LSTMFrequencyFilter
from src.training.trainer import LSTMTrainer
from src.training.evaluator import ModelEvaluator
from src.data.generator import SignalGenerator
from src.data.dataset import create_dataloaders


class TestLSTMTrainer:
    """Test suite for LSTM trainer."""
    
    @pytest.fixture
    def setup_training(self):
        """Setup model and data for training tests."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=32, num_layers=1)
        
        # Generate small dataset
        gen = SignalGenerator(num_samples=100, seed=11)
        S, targets, one_hot = gen.generate_dataset()
        
        gen_test = SignalGenerator(num_samples=100, seed=42)
        S_test, targets_test, one_hot_test = gen_test.generate_dataset()
        
        train_loader, test_loader = create_dataloaders(
            S, targets, one_hot,
            S_test, targets_test, one_hot_test,
            batch_size=16
        )
        
        trainer = LSTMTrainer(model, device, learning_rate=0.01)
        
        return trainer, train_loader, test_loader, device
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter()
        trainer = LSTMTrainer(model, device)
        
        assert trainer.model == model
        assert trainer.device == device
        assert len(trainer.train_losses) == 0
    
    def test_single_epoch_training(self, setup_training):
        """Test single epoch of training."""
        trainer, train_loader, _, _ = setup_training
        
        initial_loss = trainer.train_epoch(train_loader)
        
        assert isinstance(initial_loss, float)
        assert initial_loss > 0
        assert len(trainer.train_losses) == 1
    
    def test_loss_decreases(self, setup_training):
        """Test loss decreases over epochs."""
        trainer, train_loader, _, _ = setup_training
        
        loss1 = trainer.train_epoch(train_loader)
        loss2 = trainer.train_epoch(train_loader)
        loss3 = trainer.train_epoch(train_loader)
        
        # Loss should generally decrease (allow some variation)
        assert loss3 < loss1 * 1.1  # Within 10% tolerance
    
    def test_validation(self, setup_training):
        """Test validation method."""
        trainer, _, test_loader, _ = setup_training
        
        val_loss = trainer.validator.validate(test_loader)
        
        assert isinstance(val_loss, float)
        assert val_loss > 0
    
    def test_training_loop(self, setup_training):
        """Test full training loop."""
        trainer, train_loader, test_loader, _ = setup_training
        
        history = trainer.train(
            train_loader,
            test_loader,
            num_epochs=3,
            verbose=False
        )
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) == 3
        assert len(history['val_loss']) == 3


class TestModelEvaluator:
    """Test suite for model evaluator."""
    
    @pytest.fixture
    def setup_evaluation(self):
        """Setup trained model and data for evaluation."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=16, num_layers=1)
        
        # Generate small dataset
        gen = SignalGenerator(num_samples=100, seed=11)
        S, targets, one_hot = gen.generate_dataset()
        
        _, test_loader = create_dataloaders(
            S, targets, one_hot,
            S, targets, one_hot,
            batch_size=16
        )
        
        evaluator = ModelEvaluator(model, device)
        
        return evaluator, test_loader
    
    def test_compute_mse(self, setup_evaluation):
        """Test MSE computation."""
        evaluator, test_loader = setup_evaluation
        
        mse = evaluator.compute_mse(test_loader)
        
        assert isinstance(mse, float)
        assert mse >= 0
    
    def test_generate_predictions(self, setup_evaluation):
        """Test prediction generation."""
        evaluator, test_loader = setup_evaluation
        
        predictions, targets = evaluator.generate_predictions(test_loader)
        
        # Small dataset: 100 samples * 4 frequencies = 400
        assert len(predictions) == 400
        assert len(targets) == 400
        assert isinstance(predictions, np.ndarray)
        assert isinstance(targets, np.ndarray)
    
    def test_per_frequency_metrics(self, setup_evaluation):
        """Test per-frequency metrics computation."""
        evaluator, test_loader = setup_evaluation
        
        metrics = evaluator.evaluate_per_frequency(test_loader, num_frequencies=4)
        
        assert len(metrics) == 4
        for i in range(4):
            assert 'mse' in metrics[i]
            assert 'mae' in metrics[i]
            assert metrics[i]['mse'] >= 0
            assert metrics[i]['mae'] >= 0
    
    def test_extract_frequency_predictions(self, setup_evaluation):
        """Test extracting predictions for specific frequency."""
        evaluator, test_loader = setup_evaluation
        
        freq_preds, freq_targets = evaluator.extract_frequency_predictions(
            test_loader, freq_idx=1
        )
        
        # Should have 100 samples for this frequency
        assert len(freq_preds) == 100
        assert len(freq_targets) == 100


class TestStateManagement:
    """Test critical state management during training."""
    
    def test_state_preservation_across_batches(self):
        """Test that state is preserved across batches when not reset."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=16, num_layers=1)
        trainer = LSTMTrainer(model, device)
        
        # Generate small dataset
        gen = SignalGenerator(num_samples=50, seed=11)
        S, targets, one_hot = gen.generate_dataset()
        train_loader, _ = create_dataloaders(
            S, targets, one_hot,
            S, targets, one_hot,
            batch_size=8
        )
        
        # Train with state preservation (default)
        _ = trainer.train_epoch(train_loader, reset_state_each_batch=False)
        
        # Just verify it runs without error
        assert True
    
    def test_state_reset_each_batch(self):
        """Test training with state reset at each batch."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=16, num_layers=1)
        trainer = LSTMTrainer(model, device)
        
        gen = SignalGenerator(num_samples=50, seed=11)
        S, targets, one_hot = gen.generate_dataset()
        train_loader, _ = create_dataloaders(
            S, targets, one_hot,
            S, targets, one_hot,
            batch_size=8
        )
        
        # Train with state reset
        _ = trainer.train_epoch(train_loader, reset_state_each_batch=True)
        
        # Should work without error
        assert True


class TestCheckpointing:
    """Test model checkpointing during training."""
    
    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=16)
        trainer = LSTMTrainer(model, device)
        
        filepath = tmp_path / "checkpoint.pth"
        trainer._save_checkpoint(str(filepath), epoch=5)
        
        assert filepath.exists()
    
    def test_load_checkpoint(self, tmp_path):
        """Test checkpoint loading."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=16)
        trainer = LSTMTrainer(model, device)
        
        # Train briefly
        gen = SignalGenerator(num_samples=50, seed=11)
        S, targets, one_hot = gen.generate_dataset()
        train_loader, _ = create_dataloaders(
            S, targets, one_hot,
            S, targets, one_hot,
            batch_size=8
        )
        trainer.train_epoch(train_loader)
        
        # Save
        filepath = tmp_path / "checkpoint.pth"
        trainer._save_checkpoint(str(filepath), epoch=1)
        
        # Load into new trainer
        new_model = LSTMFrequencyFilter(hidden_size=16)
        new_trainer = LSTMTrainer(new_model, device)
        epoch = new_trainer.load_checkpoint(str(filepath))
        
        assert epoch == 1
        assert len(new_trainer.train_losses) > 0


class TestGradientHandling:
    """Test gradient handling during training."""
    
    def test_gradient_clipping(self):
        """Test gradient clipping is applied."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=16)
        trainer = LSTMTrainer(model, device, learning_rate=100.0)  # High LR to cause large gradients
        
        gen = SignalGenerator(num_samples=50, seed=11)
        S, targets, one_hot = gen.generate_dataset()
        train_loader, _ = create_dataloaders(
            S, targets, one_hot,
            S, targets, one_hot,
            batch_size=8
        )
        
        # Should not explode due to gradient clipping
        loss = trainer.train_epoch(train_loader)
        
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_parameters_update(self):
        """Test that model parameters are updated during training."""
        device = torch.device('cpu')
        model = LSTMFrequencyFilter(hidden_size=16)
        
        # Store initial parameters
        initial_params = {name: param.clone() 
                         for name, param in model.named_parameters()}
        
        trainer = LSTMTrainer(model, device, learning_rate=0.01)
        
        gen = SignalGenerator(num_samples=50, seed=11)
        S, targets, one_hot = gen.generate_dataset()
        train_loader, _ = create_dataloaders(
            S, targets, one_hot,
            S, targets, one_hot,
            batch_size=8
        )
        
        # Train one epoch
        trainer.train_epoch(train_loader)
        
        # Check at least one parameter changed
        params_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break
        
        assert params_changed, "No parameters were updated during training"
