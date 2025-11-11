# Product Requirements Document (PRD)
# LSTM Frequency Filter: Pure Signal Extraction from Noisy Mixed Signals

**Version:** 1.0  
**Last Updated:** November 11, 2025  
**Project Type:** M.Sc. Deep Learning Research Project

---

## 1. Problem Statement

### 1.1 User Problem
In signal processing applications, extracting individual frequency components from mixed, noisy signals is a fundamental challenge. Traditional methods like Fourier transforms provide frequency domain analysis but struggle with:
- Adaptive filtering of time-varying noise
- Selective extraction of specific frequencies on demand
- Learning complex noise patterns without explicit modeling

### 1.2 Project Purpose
This project implements a deep learning solution using LSTM neural networks to extract pure sinusoidal frequency components from a complex mixed signal corrupted by amplitude and phase noise that changes at every time step. The system performs **conditional regression**: given a noisy mixed signal and a frequency selector, it outputs the pure sinusoid at the requested frequency.

### 1.3 Target Users
- **Academic Researchers**: Deep learning and signal processing researchers studying RNN applications
- **Students**: M.Sc./Ph.D. students learning LSTM state management and time series modeling
- **Engineers**: Signal processing engineers exploring ML-based filtering alternatives

---

## 2. Goals and Success Metrics (KPIs)

### 2.1 Primary Goals
1. **Accurate Frequency Extraction**: Extract pure frequency components from noisy mixed signals
2. **Generalization**: Demonstrate learning of frequency structure, not noise memorization
3. **Pedagogical Value**: Illustrate LSTM state management with L=1 architecture

### 2.2 Measurable Success Metrics

| Metric | Target | Status Threshold | Actual Achievement |
|--------|--------|------------------|-------------------|
| **Test MSE** | < 0.05 | Excellent: < 0.05<br>Good: 0.05-0.10<br>Poor: > 0.10 | 0.0446 ✅ |
| **Generalization Gap** | < 0.01 | Excellent: < 0.01<br>Acceptable: 0.01-0.03<br>Overfitting: > 0.03 | 0.0024 ✅ |
| **Per-Frequency MAE** | < 0.15 | Excellent: < 0.15<br>Good: 0.15-0.25<br>Poor: > 0.25 | Max: 0.1251 ✅ |
| **Training Convergence** | < 100 epochs | Fast: < 50<br>Normal: 50-100<br>Slow: > 100 | 100 ✅ |
| **Model Efficiency** | < 100K params | Efficient: < 100K<br>Acceptable: 100-500K | ~50K ✅ |
| **Test Coverage** | ≥ 85% | Exceptional: ≥ 85%<br>Good: 70-84%<br>Basic: < 70% | Target: 85%+ |

### 2.3 Secondary Goals
- **Code Quality**: Maintainable, well-documented, modular code structure
- **Reproducibility**: Deterministic results with seed control
- **Visualization**: Comprehensive analysis with 14+ publication-quality plots
- **Educational Impact**: Clear demonstration of LSTM temporal memory

---

## 3. Functional Requirements

### 3.1 Core Functionality

#### FR-1: Signal Generation
**Priority:** Critical  
**Description:** Generate synthetic training and test datasets
- **FR-1.1:** Generate mixed noisy signal S(t) = (1/4)Σ[A_i(t)·sin(2πf_i·t + φ_i(t))]
- **FR-1.2:** Apply per-sample noise: A_i(t) ~ U(0.8, 1.2), φ_i(t) ~ U(0, 0.1π)
- **FR-1.3:** Generate pure target signals: Target_i(t) = sin(2πf_i·t)
- **FR-1.4:** Support different random seeds for train/test splits
- **FR-1.5:** Fixed parameters: 4 frequencies [1, 3, 5, 7 Hz], 10 seconds, 10,000 samples

#### FR-2: LSTM Model
**Priority:** Critical  
**Description:** Implement LSTM architecture for conditional frequency extraction
- **FR-2.1:** Input: 5-dimensional vector [S(t), C1, C2, C3, C4]
- **FR-2.2:** Architecture: 2-layer LSTM with 64 hidden units
- **FR-2.3:** Output: Single scalar (predicted pure frequency value)
- **FR-2.4:** State management: Preserve (h_t, c_t) across batches, detach for gradient control
- **FR-2.5:** Dropout: 0.2 between LSTM layers for regularization

#### FR-3: Training Pipeline
**Priority:** Critical  
**Description:** Train model with proper state management
- **FR-3.1:** L=1 training: Process one time step per batch
- **FR-3.2:** State preservation: Maintain hidden state across consecutive samples
- **FR-3.3:** Gradient detachment: Prevent backprop through entire history
- **FR-3.4:** Optimizer: Adam with learning rate 0.001
- **FR-3.5:** Loss function: Mean Squared Error (MSE)
- **FR-3.6:** Batch size: 32 samples
- **FR-3.7:** Training duration: 100 epochs with checkpointing every 20 epochs

#### FR-4: Evaluation
**Priority:** Critical  
**Description:** Comprehensive model evaluation
- **FR-4.1:** Compute overall MSE on train and test sets
- **FR-4.2:** Calculate per-frequency metrics (MSE, MAE)
- **FR-4.3:** Generate predictions for all test samples
- **FR-4.4:** Measure generalization gap (|test_MSE - train_MSE|)

#### FR-5: Visualization
**Priority:** High  
**Description:** Generate comprehensive analysis visualizations
- **FR-5.1:** Signal analysis: time domain, FFT, spectrogram, overlays
- **FR-5.2:** Training analysis: loss curves, model I/O structure
- **FR-5.3:** Prediction analysis: predictions vs actual, error distributions
- **FR-5.4:** Frequency analysis: FFT comparison, per-frequency metrics
- **FR-5.5:** All plots at 300 DPI publication quality

#### FR-6: Testing
**Priority:** High  
**Description:** Comprehensive test coverage
- **FR-6.1:** Unit tests for signal generation with noise bounds verification
- **FR-6.2:** Tests for dataset structure and one-hot encoding
- **FR-6.3:** Tests for LSTM forward pass and state management
- **FR-6.4:** Tests for training loop and gradient handling
- **FR-6.5:** Tests for evaluation metrics computation
- **FR-6.6:** Target: ≥85% code coverage

### 3.2 Configuration Management

#### FR-7: Externalized Configuration
**Priority:** High  
**Description:** No hardcoded values in source code
- **FR-7.1:** YAML configuration files for all hyperparameters
- **FR-7.2:** Environment variables for paths and system settings
- **FR-7.3:** Configuration validation on load
- **FR-7.4:** Multiple configuration profiles (default, experimental)

### 3.3 Extensibility

#### FR-8: Plugin Architecture
**Priority:** Medium  
**Description:** Support for custom extensions
- **FR-8.1:** Pluggable signal generators for different frequency patterns
- **FR-8.2:** Custom loss function support
- **FR-8.3:** Custom visualization modules
- **FR-8.4:** Clear extension points documented

---

## 4. Non-Functional Requirements

### 4.1 Performance Requirements

#### NFR-1: Training Performance
- **Training time:** < 10 minutes on CPU for 100 epochs
- **Memory usage:** < 2GB RAM during training
- **GPU acceleration:** Optional but supported

#### NFR-2: Inference Performance
- **Prediction speed:** > 1000 samples/second on CPU
- **Model size:** < 5MB saved checkpoint

### 4.2 Quality Requirements

#### NFR-3: Code Quality
- **Modularity:** All files ≤ 150 lines
- **Documentation:** Docstrings for all functions/classes with params, returns, examples
- **Style:** Consistent PEP 8 compliance
- **Testing:** ≥85% code coverage

#### NFR-4: Reproducibility
- **Deterministic:** Same results with same seeds
- **Version control:** All dependencies pinned
- **Documentation:** Complete setup instructions

### 4.3 Usability Requirements

#### NFR-5: Ease of Use
- **Installation:** One-command setup with requirements.txt
- **Execution:** Single script to run full pipeline
- **Configuration:** Human-readable YAML files
- **Troubleshooting:** Documented common issues and solutions

### 4.4 Maintainability Requirements

#### NFR-6: Maintainability
- **Modular design:** Clear separation of concerns
- **Low coupling:** Independent components
- **High cohesion:** Related functionality grouped
- **Extensibility:** Plugin architecture for customization

---

## 5. Dependencies

### 5.1 Core Dependencies
- **PyTorch ≥ 2.0.0:** Deep learning framework for LSTM implementation
- **NumPy ≥ 1.24.0:** Numerical computing for signal generation
- **SciPy ≥ 1.10.0:** Scientific computing for FFT and signal processing

### 5.2 Visualization Dependencies
- **Matplotlib ≥ 3.7.0:** Primary plotting library
- **Seaborn ≥ 0.12.0:** Statistical visualizations

### 5.3 Testing Dependencies
- **pytest ≥ 7.3.0:** Testing framework
- **pytest-cov ≥ 4.1.0:** Coverage reporting

### 5.4 Utility Dependencies
- **PyYAML ≥ 6.0:** Configuration file parsing
- **tqdm ≥ 4.65.0:** Progress bars

### 5.5 System Requirements
- **Python:** 3.8+
- **OS:** Linux, macOS, Windows
- **RAM:** 2GB minimum, 4GB recommended
- **Storage:** 100MB for code, 500MB for outputs

---

## 6. Assumptions and Constraints

### 6.1 Assumptions
1. **Fixed Frequencies:** System designed for exactly 4 frequencies [1, 3, 5, 7 Hz]
2. **Synthetic Data:** Training uses generated signals, not real-world recordings
3. **Noise Model:** Amplitude noise U(0.8, 1.2) and phase noise U(0, 0.1π) are adequate
4. **Deterministic Noise:** Different seeds produce genuinely different noise patterns
5. **L=1 Pedagogical:** Sequence length of 1 chosen for educational purposes, not efficiency

### 6.2 Constraints
1. **Computational:** Must run on standard laptop CPU (no GPU required)
2. **Time:** Training must complete within reasonable time (< 15 minutes)
3. **Memory:** Must fit in 2GB RAM for accessibility
4. **Dependencies:** Only open-source libraries, no proprietary software
5. **Scope:** M.Sc. assignment scope, not production deployment

### 6.3 Out of Scope
- Real-time signal processing
- Variable number of frequencies
- Real-world noisy audio signals
- Production deployment infrastructure
- Web interface or API server
- Mobile deployment

---

## 7. Timeline and Milestones

### Phase 1: Foundation (Completed)
- ✅ Signal generation implementation
- ✅ Dataset creation and validation
- ✅ LSTM model architecture

### Phase 2: Training (Completed)
- ✅ Training loop with state management
- ✅ Evaluation metrics implementation
- ✅ Model convergence to target MSE

### Phase 3: Analysis (Completed)
- ✅ Comprehensive visualization suite
- ✅ Per-frequency performance analysis
- ✅ Generalization verification

### Phase 4: Excellence (Current)
- 🔄 Documentation completion (PRD, Architecture, ADRs)
- 🔄 Configuration management implementation
- 🔄 Code refactoring to meet standards
- 🔄 Analysis notebook with academic rigor
- 🔄 Test coverage verification ≥85%
- 🔄 Extensibility documentation

### Phase 5: Publication Ready (Target)
- ⏳ Final quality assurance
- ⏳ All success criteria verified
- ⏳ Ready for academic submission

---

## 8. Success Criteria Summary

The project is considered successful when:

1. **✅ Performance:** Test MSE < 0.05, Generalization gap < 0.01
2. **🔄 Documentation:** Complete PRD, Architecture, ADRs, Testing, Experiments
3. **🔄 Code Quality:** All files ≤150 lines, ≥85% test coverage
4. **🔄 Configuration:** All values externalized, no hardcoded parameters
5. **🔄 Research:** Analysis notebook with academic references and LaTeX formulas
6. **🔄 Extensibility:** Plugin architecture documented with examples
7. **✅ Reproducibility:** Deterministic results, complete setup instructions
8. **✅ Visualization:** 14+ publication-quality plots

**Current Status:** 50% complete (Performance ✅, Analysis in progress 🔄)

---

## 9. Approval and Sign-off

**Document Status:** Draft v1.0  
**Review Required:** Project stakeholders, academic advisor  
**Approval Date:** Pending review

---

## Appendix A: Glossary

- **L=1:** Sequence length of 1, processing one time step per forward pass
- **MSE:** Mean Squared Error, primary loss metric
- **MAE:** Mean Absolute Error, secondary evaluation metric
- **Generalization Gap:** Difference between test and train MSE
- **State Management:** Preserving LSTM hidden state (h_t, c_t) between samples
- **One-Hot Vector:** Binary selection vector [C1, C2, C3, C4] indicating target frequency
- **Conditional Regression:** Predicting continuous output based on input condition (frequency selector)

---

*This document represents the complete product requirements for the LSTM Frequency Filter project, serving as the foundation for architecture, implementation, and evaluation.*

