"""Signal generation with per-sample noise for LSTM frequency extraction.

This module generates mixed noisy signals and pure target frequencies according to:
- Noisy signal: S(t) = (1/4)Σ[A_i(t)·sin(2πf_i·t + φ_i(t))]
- Pure targets: Target_i(t) = sin(2πf_i·t)
where noise (A_i, φ_i) changes at every sample.
"""

import numpy as np
from typing import Tuple, List


class SignalGenerator:
    """Generates mixed noisy signals and pure frequency targets for LSTM training.
    
    CRITICAL: Noise (amplitude A_i and phase φ_i) changes at EVERY sample.
    This ensures the network learns frequency structure, not noise patterns.
    """
    
    def __init__(
        self,
        frequencies: List[float] = [1.0, 3.0, 5.0, 7.0],
        time_range: Tuple[float, float] = (0.0, 10.0),
        num_samples: int = 10000,
        seed: int = 11
    ):
        """Initialize signal generator.
        
        Args:
            frequencies: List of frequencies in Hz [f1, f2, f3, f4]
            time_range: (start, end) time in seconds
            num_samples: Total number of time samples
            seed: Random seed for reproducibility
        """
        self.frequencies = frequencies
        self.time_range = time_range
        self.num_samples = num_samples
        self.seed = seed
        self.num_frequencies = len(frequencies)
        
        # Generate time array
        self.t = np.linspace(time_range[0], time_range[1], num_samples)
        
    def generate_noisy_signal(self) -> np.ndarray:
        """Generate mixed noisy signal S(t) = (1/4)Σ[A_i(t)·sin(2πf_i·t + φ_i(t))].
        
        CRITICAL: Amplitude and phase noise regenerated at EVERY sample t.
        - A_i(t) ~ Uniform(0.8, 1.2) for each sample
        - φ_i(t) ~ Uniform(0, 0.1*π) for each sample
        
        Returns:
            S: Mixed noisy signal of shape [num_samples]
        """
        np.random.seed(self.seed)
        
        # Initialize mixed signal
        S = np.zeros(self.num_samples)
        
        # For each frequency, generate noisy component
        for freq in self.frequencies:
            # Generate per-sample amplitude noise: A_i(t) ~ Uniform(0.8, 1.2)
            A_i = np.random.uniform(0.8, 1.2, size=self.num_samples)
            
            # Generate per-sample phase noise: φ_i(t) ~ Uniform(0, 0.1*π)
            phi_i = np.random.uniform(0, 0.1 * np.pi, size=self.num_samples)
            
            # Compute noisy sinusoid: A_i(t)·sin(2πf_i·t + φ_i(t))
            noisy_sinus = A_i * np.sin(2 * np.pi * freq * self.t + phi_i)
            
            # Accumulate to mixed signal
            S += noisy_sinus
        
        # Normalize by number of frequencies: S(t) = (1/4)Σ[...]
        S = S / self.num_frequencies
        
        return S
    
    def generate_pure_targets(self) -> np.ndarray:
        """Generate pure target signals Target_i(t) = sin(2πf_i·t).
        
        These are the ground truth signals without any noise.
        
        Returns:
            targets: Array of shape [num_frequencies, num_samples]
                    Each row contains pure sine wave for one frequency
        """
        targets = np.zeros((self.num_frequencies, self.num_samples))
        
        for i, freq in enumerate(self.frequencies):
            # Pure sinusoid without noise: sin(2πf_i·t)
            targets[i] = np.sin(2 * np.pi * freq * self.t)
        
        return targets
    
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate complete dataset for training/testing.
        
        Creates 40,000 rows (10,000 samples × 4 frequencies) where each row contains:
        - One sample from noisy signal S(t)
        - One-hot selection vector C indicating which frequency to extract
        - Corresponding pure target value
        
        Returns:
            S: Noisy mixed signal [num_samples]
            targets_flat: Pure targets [num_samples * num_frequencies]
            one_hot_vectors: Selection vectors [num_samples * num_frequencies, num_frequencies]
        """
        # Generate noisy mixed signal
        S = self.generate_noisy_signal()
        
        # Generate pure targets [num_frequencies, num_samples]
        targets = self.generate_pure_targets()
        
        # Create expanded dataset with one-hot encoding
        # We need 40,000 rows: for each time sample, we have 4 rows (one per frequency)
        total_samples = self.num_samples * self.num_frequencies
        
        # Expand noisy signal S: repeat each sample num_frequencies times
        S_expanded = np.repeat(S, self.num_frequencies)
        
        # Flatten targets: reshape from [num_frequencies, num_samples] to [total_samples]
        # Order: all frequencies for t=0, then all for t=1, etc.
        targets_flat = targets.T.flatten()  # Transpose then flatten for correct ordering
        
        # Create one-hot vectors for frequency selection
        one_hot_vectors = np.zeros((total_samples, self.num_frequencies))
        for i in range(total_samples):
            freq_idx = i % self.num_frequencies
            one_hot_vectors[i, freq_idx] = 1.0
        
        return S_expanded, targets_flat, one_hot_vectors
    
    def get_time_array(self) -> np.ndarray:
        """Return time array for plotting and analysis."""
        return self.t


def create_train_test_datasets(
    train_seed: int = 11,
    test_seed: int = 42
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], 
           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Create both training and test datasets with different noise seeds.
    
    CRITICAL: Train and test use SAME frequencies but DIFFERENT noise patterns.
    This tests the network's ability to generalize to unseen noise.
    
    Args:
        train_seed: Random seed for training set (default: 11 as per assignment)
        test_seed: Random seed for test set (different noise pattern)
    
    Returns:
        train_data: Tuple of (S_train, targets_train, one_hot_train)
        test_data: Tuple of (S_test, targets_test, one_hot_test)
    """
    # Generate training dataset with seed #11
    train_gen = SignalGenerator(seed=train_seed)
    train_data = train_gen.generate_dataset()
    
    # Generate test dataset with different seed
    test_gen = SignalGenerator(seed=test_seed)
    test_data = test_gen.generate_dataset()
    
    return train_data, test_data
