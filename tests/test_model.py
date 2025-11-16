"""Tests for LSTM model module."""

import pytest
import torch
import numpy as np
from src.models.lstm_filter import LSTMFrequencyFilter
from src.models.model_factory import create_model


class TestLSTMFrequencyFilter:
    """Test suite for LSTM model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = LSTMFrequencyFilter(
            input_size=5,
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        assert model.input_size == 5
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.output_size == 1
    
    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        model = LSTMFrequencyFilter(hidden_size=32, num_layers=1)
        
        # Create dummy input [batch=4, seq_len=1, features=5]
        x = torch.randn(4, 1, 5)
        
        output, (h_n, c_n) = model(x)
        
        assert output.shape == (4, 1, 1)  # [batch, seq_len, output_size]
        assert h_n.shape == (1, 4, 32)     # [num_layers, batch, hidden_size]
        assert c_n.shape == (1, 4, 32)
    
    def test_state_management(self):
        """Test hidden state preservation."""
        model = LSTMFrequencyFilter(hidden_size=32, num_layers=1)
        device = torch.device('cpu')
        
        batch_size = 2
        x1 = torch.randn(batch_size, 1, 5)
        x2 = torch.randn(batch_size, 1, 5)
        
        # First pass with no initial state
        output1, (h1, c1) = model(x1)
        
        # Second pass with state from first
        output2, (h2, c2) = model(x2, (h1, c1))
        
        # States should have changed
        assert not torch.allclose(h1, h2)
        assert not torch.allclose(c1, c2)
    
    def test_init_hidden(self):
        """Test hidden state initialization."""
        model = LSTMFrequencyFilter(hidden_size=64, num_layers=2)
        device = torch.device('cpu')
        
        h, c = model.init_hidden(batch_size=8, device=device)
        
        assert h.shape == (2, 8, 64)  # [num_layers, batch, hidden]
        assert c.shape == (2, 8, 64)
        assert torch.all(h == 0)
        assert torch.all(c == 0)
    
    def test_predict_mode(self):
        """Test prediction without gradients."""
        model = LSTMFrequencyFilter(hidden_size=32)
        x = torch.randn(2, 1, 5)
        
        output, hidden = model.predict(x)
        
        assert output.shape == (2, 1, 1)
        assert not output.requires_grad
    
    def test_different_batch_sizes(self):
        """Test model handles variable batch sizes."""
        model = LSTMFrequencyFilter(hidden_size=32)
        
        # Small batch
        x1 = torch.randn(2, 1, 5)
        output1, _ = model(x1)
        assert output1.shape == (2, 1, 1)
        
        # Large batch
        x2 = torch.randn(32, 1, 5)
        output2, _ = model(x2)
        assert output2.shape == (32, 1, 1)
    
    def test_model_parameters(self):
        """Test model has trainable parameters."""
        model = LSTMFrequencyFilter(hidden_size=32, num_layers=2)
        
        params = list(model.parameters())
        assert len(params) > 0
        
        # All parameters should require gradients
        for param in params:
            assert param.requires_grad
    
    def test_model_device_placement(self):
        """Test model can be moved to different devices."""
        model = LSTMFrequencyFilter()
        device = torch.device('cpu')
        
        model = model.to(device)
        
        # Check all parameters are on correct device
        for param in model.parameters():
            assert param.device.type == 'cpu'


class TestModelFactory:
    """Test model creation factory function."""
    
    def test_create_model_default(self):
        """Test creating model with default parameters."""
        model = create_model()
        
        assert isinstance(model, LSTMFrequencyFilter)
        assert model.input_size == 5
        assert model.output_size == 1
    
    def test_create_model_custom(self):
        """Test creating model with custom parameters."""
        model = create_model(
            hidden_size=128,
            num_layers=3,
            dropout=0.3
        )
        
        assert model.hidden_size == 128
        assert model.num_layers == 3
    
    def test_create_model_device(self):
        """Test model is placed on specified device."""
        device = torch.device('cpu')
        model = create_model(device=device)
        
        for param in model.parameters():
            assert param.device.type == 'cpu'


class TestModelSaveLoad:
    """Test model checkpointing."""
    
    def test_save_load_model(self, tmp_path):
        """Test saving and loading model."""
        from src.models.model_factory import save_model, load_model
        
        # Create and train briefly
        model = LSTMFrequencyFilter(hidden_size=32)
        
        # Save
        filepath = tmp_path / "test_model.pth"
        save_model(model, str(filepath), epoch=10)
        
        # Load
        loaded_model, epoch = load_model(str(filepath))
        
        assert epoch == 10
        assert loaded_model.hidden_size == 32
    
    def test_load_model_weights_match(self, tmp_path):
        """Test loaded model has same weights."""
        from src.models.model_factory import save_model, load_model
        
        model = LSTMFrequencyFilter(hidden_size=32)
        
        # Get initial weights
        original_weights = {name: param.clone() 
                           for name, param in model.named_parameters()}
        
        # Save and load
        filepath = tmp_path / "test_model.pth"
        save_model(model, str(filepath), epoch=1)
        loaded_model, _ = load_model(str(filepath))
        
        # Check weights match
        for name, param in loaded_model.named_parameters():
            torch.testing.assert_close(param, original_weights[name])


class TestModelOutput:
    """Test model output characteristics."""
    
    def test_output_is_regression(self):
        """Test output is continuous (regression, not classification)."""
        model = LSTMFrequencyFilter()
        x1 = torch.randn(10, 1, 5)
        x2 = torch.randn(10, 1, 5)
        
        output1, _ = model(x1)
        output2, _ = model(x2)
        
        # Output should be continuous values (not probabilities)
        # No softmax or sigmoid applied
        assert output1.dtype == torch.float32
        
        # Outputs should vary continuously with different inputs (not discrete classes)
        # Check that outputs are not all the same and vary continuously
        assert not torch.allclose(output1, output2)
        
        # Verify output layer has no activation function
        # (it's a simple Linear layer without sigmoid/softmax)
        assert isinstance(model.fc, torch.nn.Linear)
    
    def test_deterministic_output(self):
        """Test same input produces same output (when in eval mode)."""
        model = LSTMFrequencyFilter(hidden_size=32)
        model.eval()
        
        x = torch.randn(4, 1, 5)
        
        with torch.no_grad():
            output1, _ = model(x)
            output2, _ = model(x)
        
        torch.testing.assert_close(output1, output2)
    
    def test_different_inputs_different_outputs(self):
        """Test different inputs produce different outputs."""
        model = LSTMFrequencyFilter()
        
        x1 = torch.randn(2, 1, 5)
        x2 = torch.randn(2, 1, 5)
        
        output1, _ = model(x1)
        output2, _ = model(x2)
        
        assert not torch.allclose(output1, output2)


class TestStateDetachment:
    """Test critical state detachment for training."""
    
    def test_state_can_be_detached(self):
        """Test hidden state can be detached."""
        model = LSTMFrequencyFilter()
        x = torch.randn(2, 1, 5)
        
        output, (h, c) = model(x)
        
        # Detach states
        h_detached = h.detach()
        c_detached = c.detach()
        
        # Detached states should not require grad
        assert not h_detached.requires_grad
        assert not c_detached.requires_grad
        
        # Values should be the same
        torch.testing.assert_close(h, h_detached)
        torch.testing.assert_close(c, c_detached)
    
    def test_detached_state_usable_for_next_step(self):
        """Test detached state can be used for next forward pass."""
        model = LSTMFrequencyFilter()
        
        x1 = torch.randn(2, 1, 5)
        x2 = torch.randn(2, 1, 5)
        
        # First pass
        _, (h, c) = model(x1)
        
        # Detach
        h_det = h.detach()
        c_det = c.detach()
        
        # Second pass with detached state
        output2, _ = model(x2, (h_det, c_det))
        
        # Should work without error
        assert output2.shape == (2, 1, 1)


class TestModelEdgeCases:
    """Test edge cases."""
    
    def test_single_layer_lstm(self):
        """Test model with single LSTM layer."""
        model = LSTMFrequencyFilter(num_layers=1)
        x = torch.randn(4, 1, 5)
        
        output, (h, c) = model(x)
        
        assert h.shape[0] == 1  # num_layers
        assert output.shape == (4, 1, 1)
    
    def test_large_hidden_size(self):
        """Test model with large hidden size."""
        model = LSTMFrequencyFilter(hidden_size=512)
        x = torch.randn(2, 1, 5)
        
        output, _ = model(x)
        assert output.shape == (2, 1, 1)
    
    def test_zero_dropout(self):
        """Test model with zero dropout."""
        model = LSTMFrequencyFilter(dropout=0.0, num_layers=2)
        x = torch.randn(4, 1, 5)
        
        output, _ = model(x)
        assert output.shape == (4, 1, 1)
