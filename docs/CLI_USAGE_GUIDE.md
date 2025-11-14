# CLI Usage Guide
# LSTM Frequency Filter

**Complete Command-Line Interface Documentation with Examples**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Usage](#basic-usage)
3. [Advanced Options](#advanced-options)
4. [Configuration Files](#configuration-files)
5. [Example Workflows](#example-workflows)
6. [Output Interpretation](#output-interpretation)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Command (Uses All Defaults)

```bash
python train.py
```

**What This Does:**
- Loads configuration from `config/default.yaml`
- Trains for 100 epochs
- Uses seeds 11 (train) and 42 (test)
- Saves models to `outputs/models/`
- Generates 14 visualizations in `outputs/visualizations/`

**Expected Output:**
```
================================================================================
LSTM Frequency Filter - Training Pipeline
================================================================================

[INFO] Using device: cpu
[INFO] Configuration:
  - Epochs: 100
  - Batch size: 32
  - Learning rate: 0.001
  - Hidden size: 64
  - Number of layers: 2
  - Frequencies: [1.0, 3.0, 5.0, 7.0]
  - Train seed: 11
  - Test seed: 42

[INFO] Generating training dataset (seed=11)...
[INFO] Generating test dataset (seed=42)...
[INFO] Model parameters: 50,816

Epoch 1/100: 100%|██████████| 1250/1250 [00:04<00:00, 285.71 batches/s]
Train Loss: 0.4521, Val Loss: 0.4498

... (epochs 2-99) ...

Epoch 100/100: 100%|██████████| 1250/1250 [00:04<00:00, 290.15 batches/s]
Train Loss: 0.0422, Val Loss: 0.0446

[INFO] Generating visualizations...
[INFO] Creating 00_complete_overview.png...
[INFO] Creating 01_time_domain_signals.png...
... (12 more plots) ...

================================================================================
TRAINING PIPELINE COMPLETED SUCCESSFULLY!
================================================================================

[SUMMARY]
  - Trained for 100 epochs
  - Final Train MSE: 0.04223646
  - Final Test MSE: 0.04461475
  - Generated 14 visualization plots
  - Model saved to: outputs/models
  - Visualizations saved to: outputs/visualizations
  - Results saved to: outputs/results_summary.json

[NEXT STEPS]
  1. Review visualizations in outputs/visualizations
  2. Check results_summary.json for detailed metrics
  3. Run tests: pytest tests/ --cov=src

================================================================================
```

**Execution Time:** ~8 minutes on CPU, ~2 minutes on GPU

---

## Basic Usage

### Command Structure

```bash
python train.py [OPTIONS]
```

### Core Options

#### --epochs NUM_EPOCHS

Set number of training epochs.

```bash
# Train for 50 epochs (faster)
python train.py --epochs 50

# Train for 200 epochs (more thorough)
python train.py --epochs 200
```

**Expected Effect:**
- Fewer epochs: Faster training but potentially lower accuracy
- More epochs: Better convergence but diminishing returns after ~100

**Example Output Difference:**
```
# With --epochs 50:
Train Loss: 0.0512  # Slightly higher than 100 epochs

# With --epochs 100:
Train Loss: 0.0422  # Optimal convergence
```

#### --batch-size BATCH_SIZE

Set batch size for training.

```bash
# Smaller batch (more frequent updates)
python train.py --batch-size 16

# Larger batch (faster but less frequent updates)
python train.py --batch-size 64
```

**Trade-offs:**
- Batch size 16: More updates per epoch, potentially better convergence, slower
- Batch size 32: **Recommended** - balanced
- Batch size 64: Fewer updates, faster epochs, may need more total epochs

**Example Output:**
```
# With --batch-size 16:
[INFO] Batch size: 16
Epoch 1/100: 100%|██████████| 2500/2500 [00:08<00:00, 303.15 batches/s]

# With --batch-size 32 (default):
Epoch 1/100: 100%|██████████| 1250/1250 [00:04<00:00, 285.71 batches/s]

# With --batch-size 64:
Epoch 1/100: 100%|██████████| 625/625 [00:02<00:00, 275.42 batches/s]
```

#### --learning-rate LEARNING_RATE

Set optimizer learning rate.

```bash
# Conservative learning rate
python train.py --learning-rate 0.0005

# Aggressive learning rate
python train.py --learning-rate 0.002
```

**Recommended Ranges:**
- `0.0001` - Too slow, use for fine-tuning
- `0.0005` - Conservative, safe choice
- `0.001` - **Recommended** - optimal for most cases
- `0.002` - Faster convergence, may oscillate
- `0.005+` - Too high, training unstable

**Example Training Curves:**
```
# LR = 0.0001 (too slow):
Epoch 50: Loss still at 0.15

# LR = 0.001 (optimal):
Epoch 50: Loss at 0.05

# LR = 0.005 (unstable):
Epoch 50: Loss oscillates between 0.08-0.12
```

#### --hidden-size HIDDEN_SIZE

Set LSTM hidden layer size.

```bash
# Smaller model (faster, less capacity)
python train.py --hidden-size 32

# Larger model (slower, more capacity)
python train.py --hidden-size 128
```

**Model Sizes:**
```
--hidden-size 32  → ~15K parameters  → Train time: ~5 min
--hidden-size 64  → ~50K parameters  → Train time: ~8 min (default)
--hidden-size 128 → ~180K parameters → Train time: ~14 min
```

**Expected Output:**
```
# With --hidden-size 32:
[INFO] Model parameters: 15,488

# With --hidden-size 128:
[INFO] Model parameters: 180,352
```

#### --num-layers NUM_LAYERS

Set number of LSTM layers.

```bash
# Single layer (simplest)
python train.py --num-layers 1

# Three layers (more complex)
python train.py --num-layers 3
```

**Layer Complexity:**
- 1 layer: Simple patterns, fastest training
- 2 layers: **Recommended** - good balance
- 3+ layers: Complex patterns, slower, diminishing returns

---

## Advanced Options

### Device Selection

```bash
# Force CPU usage
python train.py --device cpu

# Use GPU if available
python train.py --device cuda

# Use Apple Silicon GPU
python train.py --device mps

# Auto-detect best device (default)
python train.py --device auto
```

**Auto-detection Logic:**
1. Check for CUDA GPU → use if available
2. Check for Apple Silicon (MPS) → use if available  
3. Fall back to CPU

**Example Output:**
```
[INFO] Using device: cuda
[INFO] Model parameters: 50,816
[INFO] GPU: NVIDIA GeForce RTX 3080 (10GB)
```

### Custom Random Seeds

```bash
# Different training data
python train.py --train-seed 99 --test-seed 100

# Reproduce specific experiment
python train.py --train-seed 42 --test-seed 11
```

**Use Cases:**
- Different seeds create different noise patterns
- Useful for testing generalization across multiple noise realizations
- Default (11, 42) recommended for consistency

### Custom Frequencies

```bash
# Different frequency set
python train.py --frequencies 2.0 4.0 6.0 8.0

# Fewer frequencies (faster training)
python train.py --frequencies 1.0 5.0

# More frequencies (more challenging)
python train.py --frequencies 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
```

**Note:** Changing frequencies requires model input size adjustment (automatic).

### Output Directory

```bash
# Custom output location
python train.py --output-dir /path/to/custom/outputs

# Experiment-specific directory
python train.py --output-dir outputs/experiment_001
```

### Skip Visualization

```bash
# Train only, no plots (faster)
python train.py --no-visualization
```

**Use Case:** When running multiple experiments and only need final metrics.

**Output:**
```
[INFO] Skipping visualization generation (--no-visualization flag)
  - To generate visualizations later, run without --no-visualization
```

---

## Configuration Files

### Using Configuration File

```bash
# Load from config file
python train.py --config config/default.yaml

# Load from custom config
python train.py --config my_experiment.yaml
```

### Configuration File Format

```yaml
# my_experiment.yaml
model:
  hidden_size: 128
  num_layers: 3
  dropout: 0.3

training:
  num_epochs: 150
  batch_size: 64
  learning_rate: 0.0005

data:
  train_seed: 99
  test_seed: 100
  frequencies: [2.0, 4.0, 6.0, 8.0]
```

### Merging CLI Arguments with Config

CLI arguments override config file values:

```bash
# Config file says 100 epochs, but override to 50
python train.py --config config/default.yaml --epochs 50
```

**Priority Order:**
1. CLI arguments (highest)
2. Config file
3. Default values (lowest)

---

## Example Workflows

### Workflow 1: Quick Prototype

**Goal:** Test if approach works with minimal time.

```bash
python train.py \
  --epochs 20 \
  --batch-size 64 \
  --hidden-size 32 \
  --num-layers 1 \
  --no-visualization
```

**Expected Time:** ~2 minutes  
**Expected Output:**
```
Final Train MSE: 0.0587  # Acceptable for quick test
Final Test MSE: 0.0612
```

### Workflow 2: Standard Training (Default)

**Goal:** Production-quality model with good performance.

```bash
python train.py
```

**Expected Time:** ~8 minutes  
**Expected Output:**
```
Final Train MSE: 0.0422  # Excellent
Final Test MSE: 0.0446   # Excellent generalization
```

### Workflow 3: Maximum Accuracy

**Goal:** Best possible accuracy, time is not a constraint.

```bash
python train.py \
  --epochs 150 \
  --batch-size 16 \
  --hidden-size 128 \
  --num-layers 3 \
  --learning-rate 0.0005
```

**Expected Time:** ~25 minutes  
**Expected Output:**
```
Final Train MSE: 0.0401  # Slightly better
Final Test MSE: 0.0419   # Best achievable
```

### Workflow 4: Experiment Sweep

**Goal:** Run multiple experiments with different hyperparameters.

```bash
# Experiment 1: Baseline
python train.py --output-dir outputs/exp_001_baseline

# Experiment 2: Large model
python train.py --hidden-size 128 --output-dir outputs/exp_002_large

# Experiment 3: Low LR
python train.py --learning-rate 0.0005 --output-dir outputs/exp_003_low_lr

# Experiment 4: More epochs
python train.py --epochs 200 --output-dir outputs/exp_004_long
```

Then compare `results_summary.json` from each output directory.

### Workflow 5: GPU Acceleration

**Goal:** Fast training on GPU.

```bash
python train.py \
  --device cuda \
  --epochs 100 \
  --batch-size 128  # Larger batch for GPU efficiency
```

**Expected Time:** ~2 minutes (vs 8 minutes on CPU)

---

## Output Interpretation

### Console Output Sections

#### 1. Configuration Summary

```
[INFO] Configuration:
  - Epochs: 100
  - Batch size: 32
  - Learning rate: 0.001
  - Hidden size: 64
  - Number of layers: 2
  - Frequencies: [1.0, 3.0, 5.0, 7.0]
  - Train seed: 11
  - Test seed: 42
```

**What to Check:**
- Verify parameters match your intentions
- Note the random seeds for reproducibility

#### 2. Dataset Generation

```
[INFO] Generating training dataset (seed=11)...
[INFO] Generating test dataset (seed=42)...
```

**What This Does:**
- Creates 40,000 training samples (10,000 × 4 frequencies)
- Creates 40,000 test samples with different noise

#### 3. Model Information

```
[INFO] Model parameters: 50,816
```

**Parameter Counts:**
- 50K params: Standard (hidden=64, layers=2)
- 15K params: Small (hidden=32, layers=1)
- 180K params: Large (hidden=128, layers=2)

#### 4. Training Progress

```
Epoch 1/100: 100%|██████████| 1250/1250 [00:04<00:00, 285.71 batches/s]
Train Loss: 0.4521, Val Loss: 0.4498
```

**What to Watch:**
- Loss should decrease steadily
- Val Loss close to Train Loss = good generalization
- Val Loss >> Train Loss = overfitting

**Good Training Curve:**
```
Epoch 1:    Train Loss: 0.4521, Val Loss: 0.4498  # Starting high
Epoch 20:   Train Loss: 0.0821, Val Loss: 0.0845  # Rapid descent
Epoch 50:   Train Loss: 0.0512, Val Loss: 0.0534  # Slowing down
Epoch 100:  Train Loss: 0.0422, Val Loss: 0.0446  # Converged
```

**Bad Training Curve (Overfitting):**
```
Epoch 100:  Train Loss: 0.0322, Val Loss: 0.0780  # Gap too large!
```

#### 5. Final Summary

```
[SUMMARY]
  - Trained for 100 epochs
  - Final Train MSE: 0.04223646
  - Final Test MSE: 0.04461475
  - Generated 14 visualization plots
```

**Quality Indicators:**
- MSE < 0.05: ✅ Excellent
- MSE 0.05-0.10: ✅ Good
- MSE > 0.10: ⚠️ Needs improvement

**Generalization Gap:**
- Gap = |Test MSE - Train MSE|
- Gap < 0.01: ✅ Excellent generalization
- Gap 0.01-0.03: ✅ Good generalization
- Gap > 0.03: ⚠️ Overfitting concern

### Output Files

#### results_summary.json

```json
{
  "timestamp": "2025-11-12T14:30:00.000000",
  "configuration": { ... },
  "metrics": {
    "train_mse": 0.0422,
    "test_mse": 0.0446,
    "generalization_gap": 0.0024,
    "generalizes_well": true
  },
  "per_frequency": {
    "train": { "f1_1.0Hz": {"mse": 0.0413, "mae": 0.1162}, ...},
    "test": { "f1_1.0Hz": {"mse": 0.0471, "mae": 0.1236}, ...}
  }
}
```

**How to Use:**
- Compare metrics across experiments
- Verify generalization performance
- Check per-frequency performance

#### Model Checkpoints

Location: `outputs/models/`

```
lstm_l1_epoch20.pth    # Checkpoint at epoch 20
lstm_l1_epoch40.pth    # Checkpoint at epoch 40
lstm_l1_epoch60.pth    # Checkpoint at epoch 60
lstm_l1_epoch80.pth    # Checkpoint at epoch 80
lstm_l1_epoch100.pth   # Checkpoint at epoch 100
lstm_l1_epoch100_final.pth  # Final model
```

#### Visualizations

Location: `outputs/visualizations/`

```
00_complete_overview.png       # High-level system overview
01_time_domain_signals.png     # Raw signal visualization
02_frequency_domain_fft.png    # FFT analysis
03_spectrogram.png             # Time-frequency analysis
04_overlay_signals.png         # All frequencies overlaid
05_training_samples.png        # Sample data points
06_model_io_structure.png      # Model architecture diagram
07_training_loss.png           # Training curves ← CHECK THIS FIRST
08_predictions_vs_actual.png   # Model performance ← CHECK THIS SECOND
09_error_distribution.png      # Error analysis
10_scatter_pred_vs_actual.png  # Correlation plot
11_frequency_spectrum_comparison.png  # FFT comparison
12_long_sequence_predictions.png      # Full sequence tracking
13_per_frequency_metrics.png    # Per-frequency performance
```

---

## Troubleshooting

### Issue: Training is Very Slow

**Symptoms:**
```
Epoch 1/100: 5%|▌         | 62/1250 [01:30<28:45, 0.69 batches/s]
```

**Solutions:**

1. **Use GPU:**
   ```bash
   python train.py --device cuda
   ```

2. **Reduce batch size:**
   ```bash
   python train.py --batch-size 16
   ```

3. **Reduce model size:**
   ```bash
   python train.py --hidden-size 32 --num-layers 1
   ```

### Issue: Loss Not Decreasing

**Symptoms:**
```
Epoch 50: Train Loss: 0.4213, Val Loss: 0.4198  # Still high!
```

**Solutions:**

1. **Increase learning rate:**
   ```bash
   python train.py --learning-rate 0.002
   ```

2. **Train longer:**
   ```bash
   python train.py --epochs 200
   ```

3. **Increase model capacity:**
   ```bash
   python train.py --hidden-size 128 --num-layers 2
   ```

### Issue: Overfitting (Val Loss >> Train Loss)

**Symptoms:**
```
Epoch 100: Train Loss: 0.0212, Val Loss: 0.0821  # Gap = 0.06!
```

**Solutions:**

1. **Reduce model size:**
   ```bash
   python train.py --hidden-size 32
   ```

2. **Stop earlier:**
   ```bash
   python train.py --epochs 50
   ```

3. **Increase dropout (via config file):**
   ```yaml
   model:
     dropout: 0.4  # Default is 0.2
   ```

### Issue: Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   python train.py --batch-size 16
   ```

2. **Use CPU:**
   ```bash
   python train.py --device cpu
   ```

3. **Reduce model size:**
   ```bash
   python train.py --hidden-size 32
   ```

### Issue: Results Not Reproducible

**Problem:** Running same command gives different results.

**Solution:** Verify seeds are set:
```bash
python train.py --train-seed 11 --test-seed 42
```

Also check that you're using the same Python/PyTorch versions.

---

## Advanced Tips

### 1. Monitoring Training in Real-Time

Use `tqdm` progress bars (enabled by default):
```
Epoch 42/100: 67%|██████▋   | 840/1250 [00:02<00:01, 310.15 batches/s]
```

### 2. Comparing Experiments

```bash
# Run multiple experiments
for lr in 0.0005 0.001 0.002; do
    python train.py --learning-rate $lr \
      --output-dir outputs/exp_lr_$lr \
      --no-visualization
done

# Compare results
grep "Test MSE" outputs/exp_lr_*/results_summary.json
```

### 3. Resuming Training

Currently not supported directly, but you can load a checkpoint and continue:
```python
# In custom script
from src.models.lstm_filter import load_model
model, epoch = load_model("outputs/models/lstm_l1_epoch40.pth", device)
# Continue training from epoch 40
```

### 4. Batch Processing

```bash
# Create a bash script
#!/bin/bash
for seed in 11 22 33 44; do
    python train.py --train-seed $seed \
      --output-dir outputs/exp_seed_$seed
done
```

---

## Quick Reference Card

```
# QUICK COMMAND REFERENCE

# Default run
python train.py

# Fast prototype
python train.py --epochs 20 --no-visualization

# GPU training
python train.py --device cuda

# Maximum accuracy
python train.py --epochs 150 --hidden-size 128 --learning-rate 0.0005

# Custom config
python train.py --config my_config.yaml

# Help
python train.py --help
```

---

**Last Updated:** November 12, 2025  
**Version:** 1.0

