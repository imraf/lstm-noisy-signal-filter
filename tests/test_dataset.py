"""Tests for PyTorch dataset module."""

import pytest
import torch
import numpy as np
from src.data.generator import SignalGenerator
from src.data.dataset import FrequencyDataset, create_dataloaders


class TestFrequencyDataset:
    """Test suite for FrequencyDataset class."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        gen = SignalGenerator(seed=11)
        S, targets, one_hot = gen.generate_dataset()
        return S, targets, one_hot
    
    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization."""
        S, targets, one_hot = sample_data
        dataset = FrequencyDataset(S, targets, one_hot)
        
        assert len(dataset) == 40000
        assert dataset.inputs.shape == (40000, 1, 5)  # [batch, seq_len, features]
        assert dataset.targets.shape == (40000, 1)
    
    def test_dataset_getitem(self, sample_data):
        """Test __getitem__ returns correct shapes."""
        S, targets, one_hot = sample_data
        dataset = FrequencyDataset(S, targets, one_hot)
        
        input_sample, target_sample = dataset[0]
        
        assert input_sample.shape == (1, 5)  # [seq_len=1, features=5]
        assert target_sample.shape == (1,)   # [1]
        assert isinstance(input_sample, torch.Tensor)
        assert isinstance(target_sample, torch.Tensor)
    
    def test_input_structure(self, sample_data):
        """Test input contains [S(t), C1, C2, C3, C4]."""
        S, targets, one_hot = sample_data
        dataset = FrequencyDataset(S, targets, one_hot)
        
        input_sample, _ = dataset[0]
        
        # First element should be S(t)
        assert torch.isclose(input_sample[0, 0], torch.tensor(S[0]), atol=1e-5)
        
        # Next 4 elements should be one-hot vector
        one_hot_part = input_sample[0, 1:5].numpy()
        np.testing.assert_array_almost_equal(one_hot_part, one_hot[0])
    
    def test_one_hot_preservation(self, sample_data):
        """Test one-hot vectors are correctly preserved."""
        S, targets, one_hot = sample_data
        dataset = FrequencyDataset(S, targets, one_hot)
        
        # Check first 4 samples (should cycle through all frequencies)
        for i in range(4):
            input_sample, _ = dataset[i]
            one_hot_part = input_sample[0, 1:5].numpy()
            
            # Should sum to 1
            assert np.isclose(np.sum(one_hot_part), 1.0)
            
            # Should have exactly one 1
            assert np.sum(one_hot_part == 1.0) == 1
    
    def test_target_values(self, sample_data):
        """Test target values match input targets."""
        S, targets, one_hot = sample_data
        dataset = FrequencyDataset(S, targets, one_hot)
        
        for i in range(10):  # Check first 10 samples
            _, target_sample = dataset[i]
            assert torch.isclose(target_sample[0], torch.tensor(targets[i]), atol=1e-5)
    
    def test_get_by_frequency(self, sample_data):
        """Test extracting samples for specific frequency."""
        S, targets, one_hot = sample_data
        dataset = FrequencyDataset(S, targets, one_hot)
        
        # Extract frequency 0 (f1 = 1Hz)
        freq_inputs, freq_targets = dataset.get_by_frequency(0)
        
        # Should have 10,000 samples for this frequency
        assert freq_inputs.shape == (10000, 1, 5)
        assert freq_targets.shape == (10000, 1)
        
        # All should have one-hot vector [1, 0, 0, 0]
        for i in range(10):
            one_hot_part = freq_inputs[i, 0, 1:5].numpy()
            expected = np.array([1.0, 0.0, 0.0, 0.0])
            np.testing.assert_array_almost_equal(one_hot_part, expected)
    
    def test_tensor_types(self, sample_data):
        """Test all tensors are float32."""
        S, targets, one_hot = sample_data
        dataset = FrequencyDataset(S, targets, one_hot)
        
        input_sample, target_sample = dataset[0]
        
        assert input_sample.dtype == torch.float32
        assert target_sample.dtype == torch.float32


class TestDataLoaders:
    """Test suite for DataLoader creation."""
    
    @pytest.fixture
    def train_test_data(self):
        """Generate train and test data."""
        gen_train = SignalGenerator(seed=11)
        S_train, targets_train, one_hot_train = gen_train.generate_dataset()
        
        gen_test = SignalGenerator(seed=42)
        S_test, targets_test, one_hot_test = gen_test.generate_dataset()
        
        return (S_train, targets_train, one_hot_train,
                S_test, targets_test, one_hot_test)
    
    def test_dataloader_creation(self, train_test_data):
        """Test DataLoader creation."""
        train_loader, test_loader = create_dataloaders(
            *train_test_data,
            batch_size=32
        )
        
        assert train_loader.batch_size == 32
        assert test_loader.batch_size == 32
    
    def test_dataloader_batches(self, train_test_data):
        """Test DataLoader produces correct batch shapes."""
        train_loader, test_loader = create_dataloaders(
            *train_test_data,
            batch_size=32
        )
        
        # Get first batch
        inputs, targets = next(iter(train_loader))
        
        assert inputs.shape == (32, 1, 5)  # [batch, seq_len=1, features=5]
        assert targets.shape == (32, 1)     # [batch, 1]
    
    def test_dataloader_no_shuffle_test(self, train_test_data):
        """Test test loader never shuffles."""
        _, test_loader = create_dataloaders(
            *train_test_data,
            batch_size=32,
            shuffle_train=True  # Even if train shuffles, test should not
        )
        
        # Get data twice
        batch1 = next(iter(test_loader))
        batch2 = next(iter(test_loader))
        
        # First batches should be identical (no shuffle)
        torch.testing.assert_close(batch1[0], batch2[0])
    
    def test_dataloader_iteration(self, train_test_data):
        """Test iterating through entire dataset."""
        train_loader, _ = create_dataloaders(
            *train_test_data,
            batch_size=32
        )
        
        total_samples = 0
        for inputs, targets in train_loader:
            total_samples += inputs.size(0)
        
        # Should cover all 40,000 samples
        assert total_samples == 40000
    
    def test_dataloader_custom_batch_size(self, train_test_data):
        """Test custom batch sizes."""
        train_loader, _ = create_dataloaders(
            *train_test_data,
            batch_size=64
        )
        
        inputs, _ = next(iter(train_loader))
        assert inputs.shape[0] == 64
    
    def test_last_batch_handling(self, train_test_data):
        """Test handling of last incomplete batch."""
        train_loader, _ = create_dataloaders(
            *train_test_data,
            batch_size=100  # 40000 % 100 == 0, so no incomplete batch
        )
        
        # Iterate to last batch
        for inputs, _ in train_loader:
            last_batch_size = inputs.size(0)
        
        # Last batch should be size 100
        assert last_batch_size == 100


class TestDatasetEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        S = np.array([0.5, 0.6, 0.7, 0.8])
        targets = np.array([0.1, 0.2, 0.3, 0.4])
        one_hot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        dataset = FrequencyDataset(S, targets, one_hot)
        
        assert len(dataset) == 4
        input_sample, target_sample = dataset[0]
        assert input_sample.shape == (1, 5)
    
    def test_single_sample_batch(self, train_test_data):
        """Test with batch size of 1."""
        train_loader, _ = create_dataloaders(
            *train_test_data[:3],  # Just train data
            *train_test_data[3:],   # Just test data
            batch_size=1
        )
        
        inputs, targets = next(iter(train_loader))
        assert inputs.shape == (1, 1, 5)
        assert targets.shape == (1, 1)


class TestDataPersistence:
    """Test data saving and loading."""
    
    def test_save_load_consistency(self, tmp_path):
        """Test saving and loading dataset maintains data integrity."""
        from src.data.dataset import save_dataset, load_dataset
        
        # Generate data
        gen = SignalGenerator(seed=11)
        S, targets, one_hot = gen.generate_dataset()
        
        # Save
        filepath = tmp_path / "test_dataset.pt"
        save_dataset(S, targets, one_hot, str(filepath))
        
        # Load
        S_loaded, targets_loaded, one_hot_loaded = load_dataset(str(filepath))
        
        # Verify
        np.testing.assert_array_almost_equal(S, S_loaded)
        np.testing.assert_array_almost_equal(targets, targets_loaded)
        np.testing.assert_array_almost_equal(one_hot, one_hot_loaded)
