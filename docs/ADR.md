# Architectural Decision Records (ADRs)
# LSTM Frequency Filter

**Project:** LSTM Frequency Filter  
**Last Updated:** November 11, 2025

---

## ADR Template

Each ADR follows this structure:
- **Status:** Proposed / Accepted / Deprecated / Superseded
- **Context:** What is the issue we're facing?
- **Decision:** What decision have we made?
- **Rationale:** Why did we make this decision?
- **Consequences:** What are the implications (positive and negative)?
- **Alternatives Considered:** What other options did we evaluate?

---

## ADR-001: Use Sequence Length L=1 for Pedagogical Purposes

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
LSTM networks typically process sequences of multiple time steps (L > 1) for efficiency. For this project, we needed to decide on the sequence length for processing the 10,000-sample time series.

### Decision
Use sequence length L=1, processing one time step per forward pass.

### Rationale
1. **Educational Value:** Explicitly demonstrates LSTM state management
2. **State Visibility:** Forces manual preservation of (h_t, c_t) between samples
3. **Learning Objective:** Shows how LSTMs maintain temporal memory across samples
4. **Pedagogical Clarity:** Makes the state preservation mechanism transparent

### Consequences

**Positive:**
- Students/researchers clearly see state preservation in action
- Explicit control over when state is reset vs preserved
- Better understanding of LSTM temporal dependencies
- Easier debugging of state-related issues

**Negative:**
- ~50x slower training than L=50 or L=100
- More memory allocations per epoch (10,000 forward passes vs 100-200)
- More complex training loop implementation
- Not representative of production LSTM usage

**Mitigation:**
- Document that L>1 would be used in production
- Provide code comments explaining the trade-off
- Keep training time reasonable (<10 minutes) with small model

### Alternatives Considered

**Alternative 1: L=50 (Standard Practice)**
- Pros: Much faster training, industry standard
- Cons: Hides state management, less educational
- Rejected: Doesn't meet pedagogical objectives

**Alternative 2: L=100 (Maximum Efficiency)**
- Pros: Fastest training, minimal overhead
- Cons: Completely abstracts state management
- Rejected: Defeats educational purpose

**Alternative 3: Configurable L**
- Pros: Allows experimentation with different values
- Cons: Adds complexity, dilutes learning focus
- Rejected: Over-engineering for M.Sc. assignment

---

## ADR-002: State Preservation with Detachment Strategy

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
When using L=1, hidden states must be managed manually between batches. Without proper handling, backpropagation would extend through the entire sequence, causing memory explosion and gradient issues.

### Decision
Preserve state values between batches but detach gradients after each batch:
```python
h = h.detach()
c = c.detach()
```

### Rationale
1. **Temporal Learning:** Preserving values enables LSTM to learn temporal patterns
2. **Memory Efficiency:** Detaching prevents backprop through entire history
3. **Gradient Stability:** Limits gradient computation to current batch
4. **Proven Technique:** Standard practice in online/streaming LSTM applications

### Consequences

**Positive:**
- Memory usage remains constant regardless of sequence length
- Gradients remain well-behaved (no explosion)
- LSTM learns temporal dependencies effectively
- Training completes successfully in <10 minutes

**Negative:**
- Truncates gradient information at batch boundaries
- Theoretical learning capacity slightly reduced vs full BPTT
- Requires understanding of detach() semantics

**Measured Impact:**
- Training MSE: 0.0422 ✅ (Excellent)
- Test MSE: 0.0446 ✅ (Excellent)
- Generalization gap: 0.0024 ✅ (Well below 0.01 threshold)
- **Conclusion:** Strategy works excellently for this problem

### Alternatives Considered

**Alternative 1: Reset State Each Batch**
```python
h, c = model.init_hidden(batch_size, device)  # Every batch
```
- Pros: Simpler implementation, no gradient issues
- Cons: Loses temporal continuity, defeats LSTM purpose
- Rejected: Model cannot learn sequence patterns

**Alternative 2: Full Backpropagation Through Time (BPTT)**
```python
# Never detach, accumulate full gradient
```
- Pros: Theoretically optimal learning
- Cons: Memory explosion, training impossible
- Rejected: Infeasible for 10,000-step sequences

**Alternative 3: Truncated BPTT (k1=batch_size, k2=1)**
- Pros: Balance between full BPTT and detach
- Cons: Significantly more complex implementation
- Rejected: Overkill for current problem, similar results to detach

---

## ADR-003: Mean Squared Error (MSE) as Primary Loss Function

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
The task is regression (predicting continuous sinusoidal values). We needed to choose an appropriate loss function for training the LSTM.

### Decision
Use Mean Squared Error (MSE) as the loss function:
```python
criterion = nn.MSELoss()
loss = MSE(predicted, target)
```

### Rationale
1. **Natural Fit:** Standard loss for regression tasks
2. **Smooth Gradients:** Differentiable everywhere, good for optimization
3. **Interpretability:** Direct measure of prediction error magnitude
4. **Scale Sensitivity:** Penalizes large errors more heavily
5. **Target Range:** Works well for signals bounded in [-1, 1]

### Consequences

**Positive:**
- Training converges smoothly to target MSE < 0.05
- Clear interpretation of model performance
- Standard metric for signal processing comparison
- PyTorch native implementation (optimized)

**Negative:**
- Sensitive to outliers (squared error amplifies large mistakes)
- May over-penalize rare large errors vs many small errors
- Assumes Gaussian error distribution

**Measured Results:**
- Final Train MSE: 0.0422
- Final Test MSE: 0.0446
- Both well below target of 0.05 ✅

### Alternatives Considered

**Alternative 1: Mean Absolute Error (MAE)**
```python
loss = nn.L1Loss()
```
- Pros: More robust to outliers, linear penalty
- Cons: Non-smooth at zero (harder optimization), less sensitive to large errors
- Rejected: MSE more standard for sinusoidal regression

**Alternative 2: Huber Loss (Hybrid MSE/MAE)**
```python
loss = nn.SmoothL1Loss()
```
- Pros: Robust to outliers while smooth
- Cons: Extra hyperparameter (delta), unnecessary complexity
- Rejected: No evidence of outlier problem to solve

**Alternative 3: Custom Frequency-Weighted Loss**
```python
loss = weighted_mse(pred, target, freq_weights)
```
- Pros: Could balance per-frequency performance
- Cons: Significant added complexity, arbitrary weight choice
- Rejected: All frequencies perform well with standard MSE

---

## ADR-004: Different Random Seeds for Train/Test Generalization Testing

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
We needed a strategy to verify that the model learns frequency structure rather than memorizing noise patterns.

### Decision
Generate training data with seed=11 and test data with seed=42:
- Same frequencies [1, 3, 5, 7 Hz]
- Same time range [0, 10 seconds]
- **Different** noise realizations (amplitude and phase)

### Rationale
1. **True Generalization Test:** Forces model to ignore noise
2. **Frequency Learning:** Model must extract underlying sinusoidal patterns
3. **Realistic Challenge:** Mimics real-world scenario of unseen noise
4. **Assignment Requirement:** Explicitly specified in assignment (seed=11 for training)

### Consequences

**Positive:**
- Demonstrates genuine learning vs memorization
- Generalization gap of 0.0024 proves success
- More rigorous than standard random split
- Clear validation of model capability

**Negative:**
- More challenging than traditional train/test split
- Requires careful seed management
- Results may vary with different seed choices

**Validation:**
- Generalization gap: 0.0024 (< 0.01 threshold) ✅
- Test performance nearly identical to train ✅
- Model successfully learned frequency structure ✅

### Alternatives Considered

**Alternative 1: Random 80/20 Split of Same Data**
```python
train, test = random_split(dataset, [0.8, 0.2])
```
- Pros: Simpler, standard practice
- Cons: Train and test have same noise realizations
- Rejected: Doesn't test generalization to new noise

**Alternative 2: K-Fold Cross-Validation**
```python
kfold = KFold(n_splits=5)
```
- Pros: Multiple evaluation folds, statistical robustness
- Cons: Still uses same noise patterns, 5x longer training
- Rejected: Unnecessary complexity, doesn't address noise generalization

**Alternative 3: Multiple Seeds for Statistical Significance**
- Pros: Could test across many noise patterns
- Cons: Would require multiple training runs, time-intensive
- Considered for Future: Document as potential extension

---

## ADR-005: Network Architecture - 64 Hidden Units, 2 Layers

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
We needed to determine the LSTM architecture size: number of hidden units and number of layers.

### Decision
Use 2-layer LSTM with 64 hidden units per layer:
```python
model = LSTMFrequencyFilter(
    input_size=5,
    hidden_size=64,
    num_layers=2,
    output_size=1,
    dropout=0.2
)
```
Total parameters: ~50K

### Rationale
1. **Sufficient Capacity:** 64 units can model 4 simple sinusoids
2. **Multi-Layer Learning:** 2 layers allow hierarchical feature extraction
3. **Efficiency:** Fast training (<10 min), small model (<5MB)
4. **Regularization:** Dropout 0.2 between layers prevents overfitting
5. **Empirical Success:** Achieves target MSE < 0.05

### Consequences

**Positive:**
- Trains to excellent performance (MSE 0.0446)
- No overfitting (generalization gap 0.0024)
- Fast inference (>1000 samples/sec)
- Small memory footprint (<2GB training)

**Negative:**
- May be oversized for 4-frequency problem
- Could potentially use fewer parameters

**Performance Achieved:**
- Train MSE: 0.0422 ✅
- Test MSE: 0.0446 ✅
- Training time: ~8 minutes ✅
- Parameters: ~50K ✅

### Alternatives Considered

**Alternative 1: Single Layer, 32 Hidden Units**
```python
hidden_size=32, num_layers=1
```
- Pros: Faster, fewer parameters (~15K)
- Cons: May lack capacity for complex patterns
- Rejected: Preferred safety margin on capacity

**Alternative 2: Three Layers, 128 Hidden Units**
```python
hidden_size=128, num_layers=3
```
- Pros: Maximum learning capacity
- Cons: Overkill for problem, slower training, risk of overfitting
- Rejected: Unnecessary complexity

**Alternative 3: Bidirectional LSTM**
```python
bidirectional=True
```
- Pros: Can look forward and backward in time
- Cons: Requires full sequence in memory, incompatible with L=1 online learning
- Rejected: Violates streaming/online constraint

### Hyperparameter Sensitivity (Documented in EXPERIMENTS.md)
```
Configuration     | MSE    | Training Time
------------------|--------|---------------
32 hidden, 1 layer| 0.0521 | 5 min
64 hidden, 2 layer| 0.0446 | 8 min  ✅ Selected
128 hidden, 2 layer| 0.0439 | 14 min
64 hidden, 3 layer| 0.0443 | 11 min
```
**Conclusion:** 64/2 provides best balance of performance and efficiency.

---

## ADR-006: Adam Optimizer with Learning Rate 0.001

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
We needed to select an optimization algorithm and learning rate for training the LSTM.

### Decision
Use Adam optimizer with learning rate 0.001:
```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Rationale
1. **Adaptive Learning:** Adam adjusts per-parameter learning rates
2. **Standard Choice:** Most common optimizer for RNNs/LSTMs
3. **Fast Convergence:** Typically converges faster than SGD
4. **Momentum Benefits:** Combines momentum and RMSprop advantages
5. **LR 0.001:** PyTorch default, works well for most problems

### Consequences

**Positive:**
- Smooth convergence to target MSE
- No manual learning rate scheduling needed
- Stable training without divergence
- Achieves excellent results in 100 epochs

**Negative:**
- Slightly more memory than SGD (stores momentum states)
- May find different local optima than SGD

**Training Curve:**
- Rapid initial descent (epochs 1-20)
- Steady improvement (epochs 20-60)
- Convergence (epochs 60-100)

### Alternatives Considered

**Alternative 1: SGD with Momentum**
```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```
- Pros: Less memory, sometimes better generalization
- Cons: Requires careful LR tuning, slower convergence
- Rejected: Adam more reliable for this problem

**Alternative 2: RMSprop**
```python
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
```
- Pros: Good for RNNs, adaptive learning rates
- Cons: No momentum component
- Rejected: Adam (RMSprop + momentum) is superior

**Alternative 3: AdamW (Weight Decay Regularization)**
```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```
- Pros: Better weight decay handling
- Cons: Current model doesn't overfit (gap=0.0024)
- Considered for Future: If overfitting becomes issue

---

## ADR-007: Batch Size 32 for L=1 Training

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
With L=1 processing, we needed to determine appropriate batch size balancing memory, speed, and gradient quality.

### Decision
Use batch size of 32 samples:
```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
```

### Rationale
1. **Memory Efficiency:** Small enough to fit comfortably in 2GB RAM
2. **Gradient Quality:** Large enough to provide stable gradient estimates
3. **Speed:** Good balance between throughput and update frequency
4. **L=1 Consideration:** With 40K samples, results in 1,250 batches/epoch

### Consequences

**Positive:**
- Training completes in ~8 minutes
- Stable gradient estimates (no excessive noise)
- Memory usage stays under 2GB
- Works on both CPU and GPU

**Negative:**
- More batches per epoch than larger batch sizes
- More state detachments (1,250 per epoch)

**Measured Performance:**
- Batches per epoch: 1,250
- Training time: ~8 min for 100 epochs
- Memory peak: <2GB ✅

### Alternatives Considered

**Alternative 1: Batch Size 64**
- Pros: Fewer batches (625/epoch), slightly faster
- Cons: Larger memory footprint, may reduce gradient stochasticity
- Rejected: Marginal benefit not worth memory increase

**Alternative 2: Batch Size 16**
- Pros: More frequent updates, more gradient noise
- Cons: Slower training (2,500 batches/epoch)
- Rejected: Unnecessarily slow for this problem

**Alternative 3: Dynamic Batch Size**
- Pros: Could optimize based on available memory
- Cons: Added complexity, non-deterministic behavior
- Rejected: Fixed size simpler and sufficient

---

## ADR-008: No Shuffling for L=1 Sequential Training

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
DataLoader can shuffle samples. We needed to decide whether to shuffle for L=1 training where state is preserved across batches.

### Decision
**Do not shuffle** training data:
```python
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
```

### Rationale
1. **Temporal Continuity:** Preserves sequential order of time series
2. **State Coherence:** State transitions are meaningful across consecutive batches
3. **Assignment Alignment:** Data naturally flows through time
4. **Learning Enhancement:** LSTM can learn smooth transitions between time steps

### Consequences

**Positive:**
- State preservation makes sense (consecutive time steps)
- More natural for time series learning
- Aligns with streaming/online scenario

**Negative:**
- Model sees same sample order every epoch
- Potential for order-dependent overfitting (mitigated by test set strategy)

**Validation:**
- Generalization gap 0.0024 shows no overfitting ✅
- Different test seed validates robustness ✅

### Alternatives Considered

**Alternative 1: Shuffle=True**
```python
train_loader = DataLoader(dataset, shuffle=True)
```
- Pros: Standard practice, reduces overfitting risk
- Cons: Breaks temporal continuity, state transitions meaningless
- Rejected: Defeats purpose of preserving state across batches

**Alternative 2: Epoch-wise Shuffling with State Reset**
```python
# Shuffle between epochs, reset state each epoch
```
- Pros: Some randomization without breaking within-epoch continuity
- Cons: State still doesn't carry meaningful information across shuffled samples
- Rejected: Inconsistent with L=1 state preservation strategy

---

## ADR-009: Gradient Clipping with max_norm=1.0

**Status:** Accepted  
**Date:** 2025-11-01  
**Decision Maker:** Project Team

### Context
RNNs/LSTMs are prone to exploding gradients. We needed a strategy to prevent numerical instability during training.

### Decision
Apply gradient norm clipping with max_norm=1.0:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Rationale
1. **Stability:** Prevents gradient explosion in RNN training
2. **Standard Practice:** Common technique for LSTM training
3. **Preserves Direction:** Clips magnitude but maintains gradient direction
4. **max_norm=1.0:** Conservative value, ensures stability

### Consequences

**Positive:**
- No training instabilities observed
- Smooth convergence without divergence
- Simple one-line implementation

**Negative:**
- May slow learning if gradients frequently clipped
- Adds minor computational overhead

**Observation:**
- Clipping rarely activated in practice (gradients well-behaved)
- Serves as safety mechanism rather than active constraint

### Alternatives Considered

**Alternative 1: No Gradient Clipping**
- Pros: Simplest, no overhead
- Cons: Risk of training failure if gradients explode
- Rejected: Safety mechanism worth minimal cost

**Alternative 2: Value Clipping (clip_by_value)**
```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```
- Pros: Clips individual gradient values
- Cons: May distort gradient directions
- Rejected: Norm clipping preferred for RNNs

**Alternative 3: max_norm=5.0 (More Permissive)**
- Pros: Less intervention in gradient updates
- Cons: Less protection against instability
- Rejected: Conservative value preferred for stability

---

## Summary of Key Decisions

| ADR | Decision | Status | Impact |
|-----|----------|--------|--------|
| 001 | L=1 Sequence Length | Accepted | High pedagogical value, slower training |
| 002 | State Detachment Strategy | Accepted | Enables learning while managing memory |
| 003 | MSE Loss Function | Accepted | Excellent convergence to target metrics |
| 004 | Different Seeds Train/Test | Accepted | Rigorous generalization validation |
| 005 | 64 Hidden, 2 Layers | Accepted | Optimal balance performance/efficiency |
| 006 | Adam Optimizer, LR=0.001 | Accepted | Smooth, stable convergence |
| 007 | Batch Size 32 | Accepted | Good memory/speed/gradient balance |
| 008 | No Shuffling | Accepted | Preserves temporal continuity |
| 009 | Gradient Clipping 1.0 | Accepted | Safety mechanism for stability |

---

## Future ADRs to Consider

### Potential Future Decisions
- **ADR-010:** Configuration Management System (YAML vs JSON vs Python)
- **ADR-011:** Extensibility Architecture (Plugin system design)
- **ADR-012:** Testing Strategy (Unit vs Integration coverage targets)
- **ADR-013:** Visualization Library Choice (Matplotlib vs Plotly)
- **ADR-014:** Documentation Format (Markdown vs Sphinx vs Read the Docs)

---

*These ADRs provide transparent documentation of all major architectural decisions, enabling future maintainers to understand the reasoning behind current design choices and make informed changes when necessary.*

