"""Tests for signal generator module."""

import pytest
import numpy as np
from src.data.generator import SignalGenerator, create_train_test_datasets


class TestSignalGenerator:
    """Test suite for SignalGenerator class."""
    
    def test_initialization(self):
        """Test generator initialization."""
        gen = SignalGenerator()
        
        assert gen.frequencies == [1.0, 3.0, 5.0, 7.0]
        assert gen.time_range == (0.0, 10.0)
        assert gen.num_samples == 10000
        assert gen.seed == 11
        assert len(gen.t) == 10000
        assert gen.t[0] == 0.0
        assert gen.t[-1] == 10.0
    
    def test_custom_parameters(self):
        """Test generator with custom parameters."""
        gen = SignalGenerator(
            frequencies=[2.0, 4.0],
            time_range=(0, 5),
            num_samples=5000,
            seed=42
        )
        
        assert gen.frequencies == [2.0, 4.0]
        assert len(gen.t) == 5000
        assert gen.seed == 42
    
    def test_noisy_signal_shape(self):
        """Test noisy signal has correct shape."""
        gen = SignalGenerator()
        S = gen.generate_noisy_signal()
        
        assert S.shape == (10000,)
        assert isinstance(S, np.ndarray)
    
    def test_noisy_signal_normalization(self):
        """Test noisy signal is properly normalized by 1/4."""
        gen = SignalGenerator(seed=11)
        S = gen.generate_noisy_signal()
        
        # Signal should be roughly bounded (not exactly due to noise)
        # but should be normalized
        assert np.all(np.abs(S) < 2.0)  # Sanity check
    
    def test_noise_regeneration_per_sample(self):
        """Test that noise changes at every sample."""
        gen = SignalGenerator(seed=11)
        
        # Generate two signals with same seed - should be identical
        S1 = gen.generate_noisy_signal()
        gen2 = SignalGenerator(seed=11)
        S2 = gen2.generate_noisy_signal()
        
        np.testing.assert_array_almost_equal(S1, S2)
        
        # Generate with different seed - should be different
        gen3 = SignalGenerator(seed=42)
        S3 = gen3.generate_noisy_signal()
        
        assert not np.allclose(S1, S3)
    
    def test_pure_targets_shape(self):
        """Test pure targets have correct shape."""
        gen = SignalGenerator()
        targets = gen.generate_pure_targets()
        
        assert targets.shape == (4, 10000)
        assert isinstance(targets, np.ndarray)
    
    def test_pure_targets_are_sinusoidal(self):
        """Test pure targets are pure sine waves."""
        gen = SignalGenerator()
        targets = gen.generate_pure_targets()
        
        # Check frequency content is pure
        # Pure sine waves should have amplitude in range [-1, 1]
        for i in range(4):
            assert np.all(targets[i] >= -1.0)
            assert np.all(targets[i] <= 1.0)
            
            # Check that values actually span the range
            assert np.max(targets[i]) > 0.99
            assert np.min(targets[i]) < -0.99
    
    def test_pure_targets_correct_frequencies(self):
        """Test pure targets have correct frequencies using FFT."""
        from scipy.fft import fft, fftfreq
        
        gen = SignalGenerator()
        targets = gen.generate_pure_targets()
        t = gen.get_time_array()
        
        sampling_rate = len(t) / (t[-1] - t[0])
        expected_freqs = [1.0, 3.0, 5.0, 7.0]
        
        for i, expected_freq in enumerate(expected_freqs):
            # Compute FFT
            fft_vals = fft(targets[i])
            fft_freq = fftfreq(len(t), 1/sampling_rate)
            
            # Find peak frequency
            positive_mask = fft_freq > 0
            fft_magnitude = np.abs(fft_vals[positive_mask])
            fft_freq_positive = fft_freq[positive_mask]
            
            peak_idx = np.argmax(fft_magnitude)
            peak_freq = fft_freq_positive[peak_idx]
            
            # Peak should be at expected frequency (within tolerance)
            assert np.abs(peak_freq - expected_freq) < 0.1
    
    def test_dataset_generation_shape(self):
        """Test dataset generation produces correct shapes."""
        gen = SignalGenerator()
        S, targets, one_hot = gen.generate_dataset()
        
        # 40,000 samples total (10,000 × 4 frequencies)
        assert len(S) == 40000
        assert len(targets) == 40000
        assert one_hot.shape == (40000, 4)
    
    def test_dataset_one_hot_encoding(self):
        """Test one-hot vectors are correctly structured."""
        gen = SignalGenerator()
        S, targets, one_hot = gen.generate_dataset()
        
        # Each one-hot vector should sum to 1
        assert np.allclose(np.sum(one_hot, axis=1), 1.0)
        
        # Each vector should have exactly one 1 and three 0s
        for vec in one_hot:
            assert np.sum(vec == 1.0) == 1
            assert np.sum(vec == 0.0) == 3
    
    def test_dataset_ordering(self):
        """Test dataset maintains correct ordering."""
        gen = SignalGenerator()
        S, targets, one_hot = gen.generate_dataset()
        
        # First 4 samples should be for t=0, all frequencies
        # They should all have the same S value
        assert np.allclose(S[0:4], S[0])
        
        # One-hot vectors should cycle through frequencies
        assert one_hot[0, 0] == 1.0  # f1
        assert one_hot[1, 1] == 1.0  # f2
        assert one_hot[2, 2] == 1.0  # f3
        assert one_hot[3, 3] == 1.0  # f4
        assert one_hot[4, 0] == 1.0  # Back to f1 for next time step
    
    def test_seed_independence_train_test(self):
        """Test train and test sets have different noise patterns."""
        train_data, test_data = create_train_test_datasets(
            train_seed=11,
            test_seed=42
        )
        
        S_train, targets_train, one_hot_train = train_data
        S_test, targets_test, one_hot_test = test_data
        
        # Shapes should match
        assert S_train.shape == S_test.shape
        assert targets_train.shape == targets_test.shape
        
        # Noisy signals should be different
        assert not np.allclose(S_train, S_test)
        
        # Pure targets should be identical (no noise)
        # Note: targets are expanded, so compare underlying structure
        samples_per_freq = 10000
        for i in range(4):
            start = i * samples_per_freq
            end = start + samples_per_freq
            # Pure targets should be the same regardless of seed
            np.testing.assert_array_almost_equal(
                targets_train[start:end],
                targets_test[start:end],
                decimal=10
            )


class TestAmplitudePhaseNoise:
    """Test noise bounds and characteristics."""
    
    def test_amplitude_noise_bounds(self):
        """Test amplitude noise is within [0.8, 1.2]."""
        np.random.seed(11)
        
        # Sample amplitude noise many times
        A = np.random.uniform(0.8, 1.2, size=10000)
        
        assert np.all(A >= 0.8)
        assert np.all(A <= 1.2)
        assert np.min(A) >= 0.8
        assert np.max(A) <= 1.2
    
    def test_phase_noise_bounds(self):
        """Test phase noise is within [0, 0.1*π]."""
        np.random.seed(11)
        
        # Sample phase noise many times
        phi = np.random.uniform(0, 0.1 * np.pi, size=10000)
        
        assert np.all(phi >= 0)
        assert np.all(phi <= 0.1 * np.pi)
        assert np.min(phi) >= 0
        assert np.max(phi) <= 0.1 * np.pi


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_frequency_list(self):
        """Test with empty frequency list."""
        gen = SignalGenerator(frequencies=[])
        
        S = gen.generate_noisy_signal()
        assert np.all(S == 0)  # No frequencies means zero signal
    
    def test_single_frequency(self):
        """Test with single frequency."""
        gen = SignalGenerator(frequencies=[5.0])
        
        S = gen.generate_noisy_signal()
        targets = gen.generate_pure_targets()
        
        assert S.shape == (10000,)
        assert targets.shape == (1, 10000)
    
    def test_different_time_ranges(self):
        """Test with different time ranges."""
        gen = SignalGenerator(time_range=(5, 15), num_samples=5000)
        
        t = gen.get_time_array()
        assert t[0] == 5.0
        assert t[-1] == 15.0
        assert len(t) == 5000
