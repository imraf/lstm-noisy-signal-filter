# Workflow Guide with Visual Diagrams
# LSTM Frequency Filter

**Visual guides for common workflows and system architecture**

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Data Flow Diagrams](#data-flow-diagrams)
3. [Training Workflow](#training-workflow)
4. [Testing Workflow](#testing-workflow)
5. [Experiment Workflow](#experiment-workflow)
6. [Troubleshooting Decision Trees](#troubleshooting-decision-trees)

---

## System Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LSTM Frequency Filter System                      │
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │   CONFIG     │──▶│    DATA      │──▶│    MODEL     │           │
│  │              │   │  GENERATION  │   │     LSTM     │           │
│  │ *.yaml files │   │              │   │              │           │
│  └──────────────┘   └──────────────┘   └──────────────┘           │
│         │                  │                    │                   │
│         │                  ▼                    ▼                   │
│         │         ┌──────────────┐   ┌──────────────┐             │
│         └────────▶│   TRAINING   │◀──│  EVALUATION  │             │
│                   │              │   │              │             │
│                   │  State Mgmt  │   │  MSE, MAE    │             │
│                   └──────────────┘   └──────────────┘             │
│                          │                    │                     │
│                          ▼                    ▼                     │
│                   ┌──────────────┐   ┌──────────────┐             │
│                   │ CHECKPOINTS  │   │VISUALIZATION │             │
│                   │              │   │              │             │
│                   │   *.pth      │   │  14 Plots    │             │
│                   └──────────────┘   └──────────────┘             │
│                                                                       │
└───────────────────────────────────────────────────────────────────┘
```

### Component Interaction Map

```
         User
          │
          ▼
    ┌──────────┐
    │ train.py │  Entry point
    └─────┬────┘
          │
          ▼
┌─────────────────┐
│ Configuration   │  Load config files
│ Loader          │  Resolve env vars
└────────┬────────┘
          │
          ▼
┌─────────────────┐
│ Signal          │  Generate noisy signals
│ Generator       │  Create pure targets
└────────┬────────┘
          │
          ▼
┌─────────────────┐
│ Dataset/        │  Create DataLoaders
│ DataLoader      │  Batch management
└────────┬────────┘
          │
          ▼
┌─────────────────┐
│ Model Factory   │  Create LSTM model
│                 │  Initialize weights
└────────┬────────┘
          │
          ▼
┌─────────────────┐
│ LSTM Trainer    │  Training loop
│                 │  State management
│                 │  Gradient handling
└────────┬────────┘
          │
          ├──────────┬─────────┐
          ▼          ▼         ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │Checkpoint│ │Evaluator│ │Visualize│
    │ Manager │ │         │ │ Pipeline│
    └─────────┘ └─────────┘ └─────────┘
          │          │         │
          ▼          ▼         ▼
      *.pth     metrics.json  *.png
```

---

## Data Flow Diagrams

### Training Data Flow

```
Step 1: Signal Generation
┌────────────────────────────────────┐
│  Frequencies: [1, 3, 5, 7] Hz       │
│  Time: 0-10s, 10000 samples        │
│  Seed: 11 (train) / 42 (test)     │
└──────────────┬─────────────────────┘
               │
               ▼
┌────────────────────────────────────┐
│ For each sample t:                 │
│   A_i(t) ~ U(0.8, 1.2)            │
│   φ_i(t) ~ U(0, 0.1π)             │
│   S(t) = 1/4 Σ A_i·sin(2πf_i t+φ)│
└──────────────┬─────────────────────┘
               │
               ▼
Step 2: Dataset Construction
┌────────────────────────────────────┐
│  S_expanded: [40000]               │
│  (10000 × 4 frequencies)           │
│                                    │
│  One-hot: [40000, 4]               │
│  [1,0,0,0] for f1                 │
│  [0,1,0,0] for f2                 │
│  [0,0,1,0] for f3                 │
│  [0,0,0,1] for f4                 │
│                                    │
│  Targets: [40000]                  │
│  Pure sinusoids for each freq     │
└──────────────┬─────────────────────┘
               │
               ▼
Step 3: DataLoader Batching
┌────────────────────────────────────┐
│  Batch size: 32                    │
│  Batches per epoch: 1250           │
│  Shuffle: False (sequential)      │
│                                    │
│  Input batch: [32, 1, 5]          │
│    [S(t), C1, C2, C3, C4]         │
│                                    │
│  Target batch: [32, 1]            │
│    Pure frequency values          │
└──────────────┬─────────────────────┘
               │
               ▼
Step 4: Model Processing
┌────────────────────────────────────┐
│  LSTM Forward Pass:                │
│                                    │
│  Input [32,1,5] ──┐               │
│                   │               │
│  State (h,c)──────┼──▶ LSTM ──┐  │
│  [2,32,64]        │     2layers│  │
│                   │            │  │
│  Previous state───┘            │  │
│  (preserved)                   │  │
│                                │  │
│                                ▼  │
│                              FC    │
│                             64→1   │
│                                │  │
│                                ▼  │
│                          Output    │
│                          [32,1,1]  │
│                                │  │
│                                ▼  │
│                      New State     │
│                      (detached)    │
└────────────────────────────────────┘
```

### State Management Flow (L=1)

```
Epoch Start
  │
  ▼
┌──────────────────────────────────────┐
│ Initialize State                     │
│ h, c = zeros([2, 32, 64])           │
└──────────┬───────────────────────────┘
           │
           ▼
     ┌─────────┐
     │ Batch 1 │
     └────┬────┘
          │
          ▼
┌──────────────────────────────────────┐
│ Forward: output, (h, c) = model(x)  │
│ h = h.detach()  ← CRITICAL           │
│ c = c.detach()  ← Prevent explosion  │
│ loss.backward()                      │
│ optimizer.step()                     │
└──────────┬───────────────────────────┘
           │ State preserved (values)
           │ Gradients detached
           ▼
     ┌─────────┐
     │ Batch 2 │
     └────┬────┘
          │
          ▼
┌──────────────────────────────────────┐
│ Forward with previous state          │
│ (h, c passed as input)              │
│ Temporal learning enabled           │
└──────────┬───────────────────────────┘
           │
           │
          ...
           │
           ▼
     ┌─────────┐
     │Batch 1250│
     └────┬────┘
          │
          ▼
Epoch End (state discarded)
          │
          ▼
Next Epoch (fresh state)
```

---

## Training Workflow

### Complete Training Pipeline

```
START
  │
  ├─▶ Load Configuration
  │     │
  │     ├─ config/default.yaml
  │     ├─ CLI arguments override
  │     └─ Validate parameters
  │
  ├─▶ Device Selection
  │     │
  │     ├─ Auto: CUDA? → MPS? → CPU
  │     └─ Set torch.device
  │
  ├─▶ Data Generation
  │     │
  │     ├─ Generate train (seed=11)
  │     │   └─ 40,000 samples
  │     │
  │     └─ Generate test (seed=42)
  │         └─ 40,000 samples
  │
  ├─▶ Create DataLoaders
  │     │
  │     └─ batch_size=32, shuffle=False
  │
  ├─▶ Model Creation
  │     │
  │     ├─ LSTM(input=5, hidden=64, layers=2)
  │     └─ ~50K parameters
  │
  ├─▶ Training Loop
  │     │
  │     ├─ For epochs 1 to 100:
  │     │   │
  │     │   ├─ Train epoch
  │     │   │   └─ For each batch:
  │     │   │       ├─ Forward pass
  │     │   │       ├─ Compute loss
  │     │   │       ├─ Backward pass
  │     │   │       └─ Update weights
  │     │   │
  │     │   ├─ Validate
  │     │   │   └─ Compute val loss
  │     │   │
  │     │   └─ Save checkpoint (every 20 epochs)
  │     │
  │     └─ Save final model
  │
  ├─▶ Evaluation
  │     │
  │     ├─ Compute train MSE
  │     ├─ Compute test MSE
  │     └─ Per-frequency metrics
  │
  ├─▶ Visualization Generation
  │     │
  │     └─ Generate 14 plots
  │         ├─ Signal analysis (6 plots)
  │         ├─ Training analysis (2 plots)
  │         ├─ Prediction analysis (4 plots)
  │         └─ Frequency analysis (2 plots)
  │
  └─▶ Save Results
        │
        ├─ results_summary.json
        ├─ Model checkpoints
        └─ Visualization PNGs
        
END
```

### Decision Tree: Training Success

```
Training Complete?
  │
  ├─ YES ──▶ Check Test MSE
  │            │
  │            ├─ MSE < 0.05? ──▶ YES ──▶ ✅ EXCELLENT
  │            │                           │
  │            │                           └─▶ Deploy/Use
  │            │
  │            └─ MSE >= 0.05? ──▶ Check Gap
  │                                  │
  │                                  ├─ Gap < 0.01? ──▶ ✅ GOOD
  │                                  │                   │
  │                                  │                   └─▶ May need more training
  │                                  │
  │                                  └─ Gap >= 0.01? ──▶ ⚠️ OVERFITTING
  │                                                      │
  │                                                      └─▶ Reduce model size or
  │                                                          add regularization
  │
  └─ NO ──▶ Check Error Type
             │
             ├─ OOM Error ──▶ Reduce batch_size or hidden_size
             │
             ├─ Loss NaN ──▶ Reduce learning_rate
             │
             └─ Slow/Stuck ──▶ Check data loading
                               or increase learning_rate
```

---

## Testing Workflow

### Test Execution Flow

```
START Testing
  │
  ├─▶ Environment Setup
  │     │
  │     ├─ Activate venv
  │     └─ Install pytest, pytest-cov
  │
  ├─▶ Run Tests
  │     │
  │     ├─ pytest tests/
  │     │   │
  │     │   ├─ test_data_generator.py (17 tests)
  │     │   ├─ test_dataset.py (15 tests)
  │     │   ├─ test_model.py (20 tests)
  │     │   ├─ test_training.py (14 tests)
  │     │   ├─ test_visualization.py (30 tests)
  │     │   ├─ test_config.py (17 tests)
  │     │   └─ test_pipeline.py (12 tests)
  │     │
  │     └─ Generate coverage report
  │         └─ Target: ≥ 85%
  │
  ├─▶ Analyze Results
  │     │
  │     ├─ All tests pass? ──▶ YES ──▶ ✅
  │     │                              │
  │     │                              └─▶ Check coverage
  │     │
  │     └─ Any failures? ──▶ YES ──▶ ⚠️
  │                                   │
  │                                   ├─▶ Review failure logs
  │                                   ├─▶ Fix issues
  │                                   └─▶ Re-run tests
  │
  ├─▶ Coverage Analysis
  │     │
  │     ├─ Coverage ≥ 85%? ──▶ YES ──▶ ✅ MEETS THRESHOLD
  │     │                              │
  │     │                              └─▶ Proceed to next stage
  │     │
  │     └─ Coverage < 85%? ──▶ YES ──▶ ⚠️ ADD MORE TESTS
  │                                    │
  │                                    ├─▶ Identify uncovered code
  │                                    ├─▶ Write additional tests
  │                                    └─▶ Re-run coverage
  │
  └─▶ Generate Report
        │
        ├─ HTML: outputs/coverage/htmlcov/
        ├─ Terminal summary
        └─ Update TEST_COVERAGE_REPORT.md
        
END Testing
```

### Test Categories Decision Tree

```
What to Test?
  │
  ├─▶ Core Functionality
  │     │
  │     ├─ Signal generation ──▶ test_data_generator.py
  │     ├─ Dataset loading ──▶ test_dataset.py
  │     ├─ Model forward pass ──▶ test_model.py
  │     └─ Training loop ──▶ test_training.py
  │
  ├─▶ Configuration
  │     │
  │     ├─ Config loading ──▶ test_config.py
  │     ├─ Validation ──▶ test_config.py
  │     └─ Env vars ──▶ test_config.py
  │
  ├─▶ Integration
  │     │
  │     └─ End-to-end pipeline ──▶ test_pipeline.py
  │
  └─▶ Edge Cases
        │
        ├─ Noise bounds
        ├─ State management
        ├─ Invalid inputs
        └─ Batch handling
```

---

## Experiment Workflow

### Running Multiple Experiments

```
EXPERIMENT PLANNING
  │
  ├─▶ Define Variables
  │     │
  │     ├─ Learning rates: [0.0005, 0.001, 0.002]
  │     ├─ Hidden sizes: [32, 64, 128]
  │     └─ Layers: [1, 2, 3]
  │
  ├─▶ Create Experiment Matrix
  │     │
  │     └─ Total: 3×3×3 = 27 experiments
  │
  ├─▶ Setup Output Directories
  │     │
  │     └─ outputs/
  │         ├─ exp_lr0.0005_h32_l1/
  │         ├─ exp_lr0.0005_h32_l2/
  │         ├─ ...
  │         └─ exp_lr0.002_h128_l3/
  │
  ├─▶ Run Experiments
  │     │
  │     └─ For each configuration:
  │         │
  │         ├─ python train.py \
  │         │   --learning-rate $lr \
  │         │   --hidden-size $h \
  │         │   --num-layers $l \
  │         │   --output-dir outputs/exp_...
  │         │   --no-visualization  # Speed up
  │         │
  │         └─ Log results
  │
  ├─▶ Collect Results
  │     │
  │     └─ For each experiment:
  │         │
  │         ├─ Extract test_mse from results_summary.json
  │         ├─ Extract train_mse
  │         └─ Calculate generalization gap
  │
  ├─▶ Analyze Results
  │     │
  │     ├─ Create comparison table
  │     ├─ Identify best configuration
  │     └─ Generate comparative plots
  │
  └─▶ Document Findings
        │
        └─ Update EXPERIMENTS.md with:
            ├─ Configuration table
            ├─ Results analysis
            └─ Recommendations
```

### Experiment Results Table Format

```
┌─────┬────┬────┬────┬──────────┬──────────┬─────┬────────┐
│ Exp │ LR │ Hidden│Layers│Train MSE│Test MSE│ Gap │ Time   │
├─────┼────┼────┼────┼──────────┼──────────┼─────┼────────┤
│ 001 │.001│ 64 │ 2  │ 0.0422   │ 0.0446   │.0024│ 8 min  │ ✅ BEST
│ 002 │.001│ 32 │ 2  │ 0.0521   │ 0.0543   │.0022│ 5 min  │
│ 003 │.001│128 │ 2  │ 0.0439   │ 0.0461   │.0022│ 14 min │
│ 004 │.002│ 64 │ 2  │ 0.0459   │ 0.0483   │.0024│ 8 min  │
│ ... │... │... │... │   ...    │   ...    │ ... │  ...   │
└─────┴────┴────┴────┴──────────┴──────────┴─────┴────────┘
```

---

## Troubleshooting Decision Trees

### Issue: Poor Performance

```
Test MSE > 0.10?
  │
  ├─ YES ──▶ Check Training Loss
  │            │
  │            ├─ Train Loss > 0.10? ──▶ Model Not Learning
  │            │                          │
  │            │                          ├─▶ Increase model size
  │            │                          ├─▶ Increase learning rate
  │            │                          └─▶ Train longer
  │            │
  │            └─ Train Loss < 0.05? ──▶ Overfitting
  │                                       │
  │                                       ├─▶ Reduce model size
  │                                       ├─▶ Add regularization
  │                                       └─▶ Get more data
  │
  └─ NO ──▶ Performance acceptable, check details
```

### Issue: Training Problems

```
Training Stuck?
  │
  ├─▶ Loss Not Decreasing?
  │     │
  │     ├─ Loss constant? ──▶ Learning rate too low
  │     │                     └─▶ Increase LR to 0.002
  │     │
  │     └─ Loss oscillating? ──▶ Learning rate too high
  │                              └─▶ Decrease LR to 0.0005
  │
  ├─▶ Training Very Slow?
  │     │
  │     ├─ Using CPU? ──▶ Switch to GPU (--device cuda)
  │     │
  │     └─ Using GPU? ──▶ Increase batch size
  │
  └─▶ Memory Error?
        │
        ├─▶ Reduce batch size
        ├─▶ Reduce model size
        └─▶ Use CPU
```

### Issue: Configuration Problems

```
Config Error?
  │
  ├─▶ File Not Found?
  │     │
  │     └─▶ Check path: config/default.yaml exists?
  │
  ├─▶ Validation Error?
  │     │
  │     └─▶ Check required fields:
  │         ├─ data.train_seed
  │         ├─ model.hidden_size
  │         └─ training.learning_rate
  │
  └─▶ Invalid Values?
        │
        └─▶ Check ranges:
            ├─ learning_rate > 0
            ├─ batch_size > 0
            └─ hidden_size > 0
```

---

## Quick Reference Workflows

### 1. First-Time Setup

```
1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Run default training
5. Check results

┌──────────────────────┐
│ git clone <repo>     │
│ cd lstm-...          │
│ python3 -m venv venv │
│ source venv/bin/...  │
│ pip install -r req...│
│ python train.py      │
│ open outputs/vis/... │
└──────────────────────┘
```

### 2. Quick Experiment

```
1. Modify one parameter
2. Run with custom output dir
3. Compare results

┌──────────────────────────────────────┐
│ python train.py \                    │
│   --learning-rate 0.002 \            │
│   --output-dir outputs/exp_lr002 \   │
│   --no-visualization                 │
│                                      │
│ diff outputs/results_summary.json \ │
│      outputs/exp_lr002/results_...  │
└──────────────────────────────────────┘
```

### 3. Testing Changes

```
1. Make code changes
2. Run tests
3. Check coverage
4. Fix issues

┌──────────────────────────────────────┐
│ # Edit code                          │
│ pytest tests/ -v                     │
│ pytest tests/ --cov=src              │
│ # Fix any failures                   │
│ pytest tests/ --cov=src              │
└──────────────────────────────────────┘
```

---

**Last Updated:** November 12, 2025  
**Version:** 1.0

