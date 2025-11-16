# Extensibility Guide
# LSTM Frequency Filter Plugin Architecture

**Version:** 1.0  
**Last Updated:** November 11, 2025

---

## Overview

The LSTM Frequency Filter is designed with extensibility in mind, allowing researchers and developers to customize key components without modifying core code. This document describes extension points and provides examples.

---

## Extension Points

### 1. Custom Signal Generators

**Purpose:** Generate different signal types or noise models

**Base Class:** `SignalGenerator` in `src/data/generator.py`

**Extension Methods:**
- `generate_noisy_signal()`: Override to implement custom noise model
- `generate_pure_targets()`: Override for different target signals

**Example Use Cases:**
- Real-world audio signals
- Different noise distributions (Gaussian, Laplacian, etc.)
- Variable number of frequencies
- Time-varying frequencies (chirps)

### 2. Custom Loss Functions

**Purpose:** Experiment with different optimization objectives

**Current:** MSE loss (`nn.MSELoss`)

**Extension Method:** Pass custom loss function to trainer

**Example Use Cases:**
- MAE for robustness to outliers
- Huber loss for combined MSE/MAE
- Custom frequency-weighted loss
- Perceptual loss functions

### 3. Custom Visualization Modules

**Purpose:** Add new analysis plots

**Location:** `src/visualization/`

**Extension Method:** Create new plot functions following existing patterns

**Example Use Cases:**
- Interactive Plotly dashboards
- 3D visualizations
- Animation of training progress
- Comparison plots across experiments

### 4. Custom Metrics

**Purpose:** Additional evaluation criteria

**Location:** `src/training/evaluator.py`

**Extension Method:** Add methods to `ModelEvaluator` class

**Example Use Cases:**
- Frequency-domain metrics
- Phase accuracy
- Signal-to-noise ratio
- Correlation coefficients

### 5. Custom Optimizers

**Purpose:** Test different optimization algorithms

**Current:** Adam optimizer

**Extension Method:** Pass custom optimizer to trainer

**Example Use Cases:**
- SGD with momentum
- RMSprop
- AdamW with weight decay
- Custom learning rate schedulers

---

## Example 1: Custom Signal Generator

### Gaussian Noise Generator

```python
# examples/custom_gaussian_generator.py

import numpy as np
from src.data.generator import SignalGenerator


class GaussianNoiseGenerator(SignalGenerator):
    """Signal generator with Gaussian noise instead of uniform."""
    
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
        """Generate signal with Gaussian amplitude and phase noise."""
        np.random.seed(self.seed)
        S = np.zeros(self.num_samples)
        
        for freq in self.frequencies:
            # Gaussian amplitude noise: mean=1.0, std=noise_std
            A_i = np.random.normal(1.0, self.noise_std, size=self.num_samples)
            
            # Gaussian phase noise: mean=0, std=noise_std*pi
            phi_i = np.random.normal(0, self.noise_std * np.pi, size=self.num_samples)
            
            noisy_sinus = A_i * np.sin(2 * np.pi * freq * self.t + phi_i)
            S += noisy_sinus
        
        S = S / self.num_frequencies
        return S


# Usage
if __name__ == "__main__":
    gen = GaussianNoiseGenerator(noise_std=0.15)
    S = gen.generate_noisy_signal()
    targets = gen.generate_pure_targets()
    
    print(f"Signal shape: {S.shape}")
    print(f"Signal range: [{S.min():.3f}, {S.max():.3f}]")
```

### Variable Frequency Generator

```python
# examples/custom_variable_freq_generator.py

import numpy as np
from src.data.generator import SignalGenerator


class VariableFrequencyGenerator(SignalGenerator):
    """Generator supporting variable number of frequencies."""
    
    def __init__(
        self,
        frequencies: list,  # Can be any length
        time_range: tuple = (0.0, 10.0),
        num_samples: int = 10000,
        seed: int = 11
    ):
        super().__init__(frequencies, time_range, num_samples, seed)
    
    def generate_dataset(self):
        """Generate dataset with dynamic one-hot size."""
        S = self.generate_noisy_signal()
        targets = self.generate_pure_targets()
        
        total_samples = self.num_samples * self.num_frequencies
        S_expanded = np.repeat(S, self.num_frequencies)
        targets_flat = targets.T.flatten()
        
        # Dynamic one-hot vector size
        one_hot_vectors = np.zeros((total_samples, self.num_frequencies))
        for i in range(total_samples):
            freq_idx = i % self.num_frequencies
            one_hot_vectors[i, freq_idx] = 1.0
        
        return S_expanded, targets_flat, one_hot_vectors


# Usage: Support 8 frequencies
if __name__ == "__main__":
    gen = VariableFrequencyGenerator(
        frequencies=[1, 2, 3, 4, 5, 6, 7, 8]
    )
    S, targets, one_hot = gen.generate_dataset()
    
    print(f"Frequencies: {len(gen.frequencies)}")
    print(f"One-hot shape: {one_hot.shape}")  # [80000, 8]
    print(f"Input dimension: {one_hot.shape[1] + 1}")  # 9 = S + 8 one-hot
```

---

## Example 2: Custom Loss Function

### Frequency-Weighted Loss

```python
# examples/custom_loss.py

import torch
import torch.nn as nn


class FrequencyWeightedLoss(nn.Module):
    """Weighted MSE loss with per-frequency weights."""
    
    def __init__(self, weights: list = [1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self.weights = torch.tensor(weights)
    
    def forward(self, predictions, targets, freq_indices):
        """
        Args:
            predictions: Model outputs [N]
            targets: Ground truth [N]
            freq_indices: Frequency index for each sample [N]
        
        Returns:
            Weighted MSE loss
        """
        squared_errors = (predictions - targets) ** 2
        
        # Apply weights based on frequency
        weights = self.weights[freq_indices]
        weighted_errors = squared_errors * weights
        
        return weighted_errors.mean()


# Usage in training
if __name__ == "__main__":
    from src.training.trainer import LSTMTrainer
    from src.models.lstm_filter import create_model
    import torch
    
    model = create_model()
    device = torch.device('cpu')
    
    # Create trainer with custom loss
    trainer = LSTMTrainer(model, device)
    trainer.criterion = FrequencyWeightedLoss(
        weights=[1.5, 1.0, 1.0, 0.8]  # Emphasize low frequencies
    )
    
    # Train normally
    # history = trainer.train(...)
```

### Huber Loss (Robust to Outliers)

```python
# examples/robust_loss.py

import torch
import torch.nn as nn


def create_trainer_with_huber_loss(model, device, delta=1.0):
    """Create trainer with Huber loss instead of MSE."""
    from src.training.trainer import LSTMTrainer
    
    trainer = LSTMTrainer(model, device)
    trainer.criterion = nn.SmoothL1Loss(beta=delta)
    
    return trainer


# Usage
if __name__ == "__main__":
    from src.models.lstm_filter import create_model
    import torch
    
    model = create_model()
    device = torch.device('cpu')
    
    # Huber loss is more robust to outliers
    trainer = create_trainer_with_huber_loss(model, device, delta=0.5)
```

---

## Example 3: Custom Visualization

### Interactive Plotly Dashboard

```python
# examples/interactive_viz.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_interactive_dashboard(
    t, predictions, targets, frequencies, save_path
):
    """Create interactive Plotly dashboard."""
    samples_per_freq = len(predictions) // len(frequencies)
    
    fig = make_subplots(
        rows=len(frequencies), cols=1,
        subplot_titles=[f'Frequency {f}Hz' for f in frequencies]
    )
    
    for i, freq in enumerate(frequencies):
        start_idx = i * samples_per_freq
        end_idx = start_idx + samples_per_freq
        
        freq_preds = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        # Add traces
        fig.add_trace(
            go.Scatter(x=t, y=freq_targets, name='Target', 
                      line=dict(color='green')),
            row=i+1, col=1
        )
        fig.add_trace(
            go.Scatter(x=t, y=freq_preds, name='Prediction',
                      line=dict(color='red', dash='dash')),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=300 * len(frequencies),
        title_text="Interactive Frequency Extraction Dashboard",
        showlegend=True
    )
    
    fig.write_html(save_path)
    print(f"Interactive dashboard saved to {save_path}")


# Usage
if __name__ == "__main__":
    from src.data.generator import SignalGenerator
    from src.training.evaluator import ModelEvaluator
    
    # Assuming you have trained model and evaluator
    # evaluator = ModelEvaluator(model, device)
    # predictions, targets = evaluator.generate_predictions(test_loader)
    
    # gen = SignalGenerator(seed=42)
    # t = gen.get_time_array()
    # frequencies = [1.0, 3.0, 5.0, 7.0]
    
    # create_interactive_dashboard(
    #     t, predictions, targets, frequencies,
    #     "outputs/interactive_dashboard.html"
    # )
```

---

## Example 4: Custom Metric

### Signal-to-Noise Ratio (SNR)

```python
# examples/custom_metric.py

import numpy as np
from src.training.evaluator import ModelEvaluator


class ExtendedEvaluator(ModelEvaluator):
    """Extended evaluator with additional metrics."""
    
    def compute_snr(self, data_loader):
        """Compute Signal-to-Noise Ratio.
        
        SNR = 10 * log10(signal_power / noise_power)
        """
        predictions, targets = self.generate_predictions(data_loader)
        
        signal_power = np.mean(targets ** 2)
        noise = predictions - targets
        noise_power = np.mean(noise ** 2)
        
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        return snr_db
    
    def compute_correlation(self, data_loader):
        """Compute correlation coefficient."""
        predictions, targets = self.generate_predictions(data_loader)
        correlation = np.corrcoef(predictions, targets)[0, 1]
        return correlation
    
    def compute_rmse(self, data_loader):
        """Compute Root Mean Squared Error."""
        mse = self.compute_mse(data_loader)
        return np.sqrt(mse)


# Usage
if __name__ == "__main__":
    # Assuming trained model
    # model = ...
    # device = ...
    # test_loader = ...
    
    # evaluator = ExtendedEvaluator(model, device)
    # snr = evaluator.compute_snr(test_loader)
    # corr = evaluator.compute_correlation(test_loader)
    # rmse = evaluator.compute_rmse(test_loader)
    
    # print(f"SNR: {snr:.2f} dB")
    # print(f"Correlation: {corr:.4f}")
    # print(f"RMSE: {rmse:.6f}")
```

---

## Example 5: Custom Optimizer

### SGD with Learning Rate Scheduling

```python
# examples/custom_optimizer.py

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from src.training.trainer import LSTMTrainer


def create_trainer_with_sgd(model, device, lr=0.01, momentum=0.9):
    """Create trainer with SGD optimizer and LR scheduler."""
    trainer = LSTMTrainer(model, device)
    
    # Replace Adam with SGD
    trainer.optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=1e-4
    )
    
    # Add learning rate scheduler
    trainer.scheduler = StepLR(
        trainer.optimizer,
        step_size=30,  # Decay every 30 epochs
        gamma=0.5      # Multiply LR by 0.5
    )
    
    return trainer


# Modified training loop with scheduler
def train_with_scheduler(trainer, train_loader, val_loader, num_epochs):
    """Training loop that applies LR scheduling."""
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        
        # Step scheduler
        if hasattr(trainer, 'scheduler'):
            trainer.scheduler.step()
            current_lr = trainer.scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")


# Usage
if __name__ == "__main__":
    from src.models.lstm_filter import create_model
    import torch
    
    model = create_model()
    device = torch.device('cpu')
    
    trainer = create_trainer_with_sgd(model, device, lr=0.01, momentum=0.9)
    # train_with_scheduler(trainer, train_loader, val_loader, 100)
```

---

## Integration Patterns

### Pattern 1: Drop-in Replacement

For components with same interface:

```python
# Original
from src.data.generator import SignalGenerator
gen = SignalGenerator(seed=11)

# Custom drop-in
from examples.custom_gaussian_generator import GaussianNoiseGenerator
gen = GaussianNoiseGenerator(seed=11, noise_std=0.15)  # Same interface

# Rest of code unchanged
S, targets, one_hot = gen.generate_dataset()
```

### Pattern 2: Subclass Extension

For adding new functionality:

```python
# Original evaluator
from src.training.evaluator import ModelEvaluator
evaluator = ModelEvaluator(model, device)

# Extended evaluator with new methods
from examples.custom_metric import ExtendedEvaluator
evaluator = ExtendedEvaluator(model, device)

# Use original methods
mse = evaluator.compute_mse(test_loader)

# Use new methods
snr = evaluator.compute_snr(test_loader)
```

### Pattern 3: Configuration-Based

For runtime selection:

```python
# config.yaml
generator:
  type: "gaussian"  # or "uniform", "custom"
  params:
    noise_std: 0.15

# Factory function
def create_generator(config):
    gen_type = config['generator']['type']
    params = config['generator']['params']
    
    if gen_type == 'uniform':
        return SignalGenerator(**params)
    elif gen_type == 'gaussian':
        return GaussianNoiseGenerator(**params)
    else:
        raise ValueError(f"Unknown generator: {gen_type}")

# Usage
gen = create_generator(config)
```

---

## Best Practices

### 1. Maintain Interface Compatibility

```python
# Good: Same interface as base class
class CustomGenerator(SignalGenerator):
    def generate_noisy_signal(self) -> np.ndarray:
        # Custom implementation
        pass

# Bad: Changed interface
class BadGenerator(SignalGenerator):
    def generate_noisy_signal(self, extra_param):  # New required param
        pass
```

### 2. Document Extensions

```python
class CustomGenerator(SignalGenerator):
    """Custom generator with Gaussian noise.
    
    Extends SignalGenerator to use Gaussian instead of uniform noise.
    
    Args:
        noise_std: Standard deviation of Gaussian noise
        
    Example:
        >>> gen = CustomGenerator(noise_std=0.1)
        >>> S = gen.generate_noisy_signal()
    """
```

### 3. Provide Examples

Include runnable examples in docstrings or separate example files.

### 4. Test Extensions

```python
# tests/test_custom_generator.py
def test_custom_generator():
    gen = CustomGenerator(noise_std=0.1)
    S = gen.generate_noisy_signal()
    assert S.shape == (10000,)
    assert np.abs(S).max() < 2.0  # Reasonable bounds
```

---

## Contributing Extensions

If you develop a useful extension, consider contributing it back:

1. **Create example file** in `examples/`
2. **Add tests** in `tests/`
3. **Document in this file** with example usage
4. **Submit pull request** with clear description

---

## Extension Registry

Maintain a list of community extensions:

| Extension | Purpose | Author | Link |
|-----------|---------|--------|------|
| GaussianNoiseGen | Gaussian noise model | Core | `examples/` |
| VariableFreqGen | Variable # frequencies | Core | `examples/` |
| InteractiveDash | Plotly dashboard | Core | `examples/` |
| SNRMetric | SNR evaluation | Core | `examples/` |

---

*This extensibility guide enables researchers to customize the LSTM Frequency Filter for their specific needs while maintaining code quality and compatibility.*

