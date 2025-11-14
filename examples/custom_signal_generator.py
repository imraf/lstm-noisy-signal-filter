"""Example: Custom Signal Generator with Gaussian Noise.

This example demonstrates how to extend the SignalGenerator class
to implement a different noise model.
"""

import numpy as np
from src.data.generator import SignalGenerator


class GaussianNoiseGenerator(SignalGenerator):
    """Signal generator with Gaussian noise instead of uniform noise.
    
    Args:
        frequencies: List of frequencies in Hz
        time_range: (start, end) time in seconds
        num_samples: Total number of time samples
        seed: Random seed for reproducibility
        noise_std: Standard deviation for Gaussian noise
    """
    
    def __init__(
        self,
        frequencies: list = [1.0, 3.0, 5.0, 7.0],
        time_range: tuple = (0.0, 10.0),
        num_samples: int = 10000,
        seed: int = 11,
        noise_std: float = 0.1
    ):
        super().__init__(frequencies, time_range, num_samples, seed)
        self.noise_std = noise_std
    
    def generate_noisy_signal(self) -> np.ndarray:
        """Generate signal with Gaussian amplitude and phase noise.
        
        Instead of uniform noise U(0.8, 1.2) and U(0, 0.1π),
        uses Gaussian noise N(1.0, noise_std) and N(0, noise_std*π).
        
        Returns:
            S: Mixed noisy signal of shape [num_samples]
        """
        np.random.seed(self.seed)
        S = np.zeros(self.num_samples)
        
        for freq in self.frequencies:
            A_i = np.random.normal(1.0, self.noise_std, size=self.num_samples)
            phi_i = np.random.normal(0, self.noise_std * np.pi, size=self.num_samples)
            
            noisy_sinus = A_i * np.sin(2 * np.pi * freq * self.t + phi_i)
            S += noisy_sinus
        
        S = S / self.num_frequencies
        return S


def main():
    """Demonstrate custom generator usage."""
    print("=" * 80)
    print("Custom Signal Generator Example")
    print("=" * 80)
    
    print("\n1. Creating generators with different noise models...")
    
    uniform_gen = SignalGenerator(seed=42)
    gaussian_gen = GaussianNoiseGenerator(seed=42, noise_std=0.15)
    
    S_uniform = uniform_gen.generate_noisy_signal()
    S_gaussian = gaussian_gen.generate_noisy_signal()
    
    print(f"\nUniform Noise Signal:")
    print(f"  Range: [{S_uniform.min():.4f}, {S_uniform.max():.4f}]")
    print(f"  Mean: {S_uniform.mean():.4f}")
    print(f"  Std: {S_uniform.std():.4f}")
    
    print(f"\nGaussian Noise Signal:")
    print(f"  Range: [{S_gaussian.min():.4f}, {S_gaussian.max():.4f}]")
    print(f"  Mean: {S_gaussian.mean():.4f}")
    print(f"  Std: {S_gaussian.std():.4f}")
    
    print("\n2. Generating complete dataset...")
    S, targets, one_hot = gaussian_gen.generate_dataset()
    
    print(f"\nDataset shapes:")
    print(f"  S: {S.shape}")
    print(f"  Targets: {targets.shape}")
    print(f"  One-hot: {one_hot.shape}")
    
    print("\n3. Usage in training pipeline...")
    print("   Simply replace SignalGenerator with GaussianNoiseGenerator:")
    print("   ")
    print("   # Original:")
    print("   # gen = SignalGenerator(seed=11)")
    print("   ")
    print("   # Custom:")
    print("   # gen = GaussianNoiseGenerator(seed=11, noise_std=0.15)")
    print("   ")
    print("   # Rest of code unchanged!")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

