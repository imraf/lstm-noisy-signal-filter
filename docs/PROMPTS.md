# Project Prompts

This document contains the original prompts used to develop this LSTM Frequency Filter project.

---

## PROMPT #1: Initial Project Specification

(Note: This prompt was hand-written by us before the assignment PDF was available, and was then converted to markdown.)

We are going to train an LSTM model which is going to filter frequencies from a signal. **Important:** We want to create visual representations for our frequencies, main signal, model inputs, model outputs, and other beneficial visual aids.

### Frequency Generation

Create 4 different frequencies, and name them `f1`, `f2`, `f3`, and `f4`.

For every frequency `fi`, we will introduce noise such that:

```
f'i(t) = Ai(t) · sin(2π · fi · t + φi(t))
```

Where:
- `Ai(t)` uniformly varies in the range `(0.8, 1.2)`
- `φi(t)` uniformly varies in the range `(0, 0.1π)`

### Signal Composition

The signal will be the sum of these noisy frequencies:

```
S(t) = Σi f'i(t) = Σi Ai(t) · sin(2π · fi · t + φi(t))
```

### Step One: Create Sample Dataset

Suggest a long interval over `t` in which you will sample **10,000 values** for the clean frequencies `f1(t)`, `f2(t)`, `f3(t)`, `f4(t)` as well as the joint signal `S(t)`.

The dataset will be a table of this form:

| Sample | t value | f1(t) | f2(t) | f3(t) | f4(t) | S(t) |
|--------|---------|-------|-------|-------|-------|------|
| ...    | ...     | ...   | ...   | ...   | ...   | ...  |

To train the model, you will have to include the selector `c`, which will be in the form of a **one-hot vector**:
- `[1, 0, 0, 0]`
- `[0, 1, 0, 0]`
- `[0, 0, 1, 0]`
- `[0, 0, 0, 1]`

### Step Two: Training Setup

To train the model, use the data above to create training samples which will allow the model to filter the selected frequency `fi(t)` from the sum and selector: `ci & S(t)`. Compare the result to the actual `fi(t)` and suggest a good error and loss function for training.

**The input for the model will be:**
- The joint signal `S(t)` with a selector

**The model should output:**
- The frequency selected by `c`

### Step Three: Implementation and Training

1. Research Python frameworks offering LSTM code and choose the best framework for training this model which will be easiest to implement.

2. Research similar problems and suggest the best hyperparameters for this model.

3. Finally, train the model, assess its accuracy and create the required graphs.

---

## PROMPT #2: Project Planning and Documentation

(Note: This was based on the assignment PDF which we converted to markdown.)

Convert the assignment requirements into structured project documentation:

### Task 1: Requirements Analysis

1. **Extract and convert** the assignment PDF into markdown format, preserving:
   - All technical specifications
   - Mathematical formulas and notation
   - Evaluation criteria and grading rubrics
   - Success metrics and KPIs
   - Timeline and deliverables

2. **Create a comprehensive Product Requirements Document (PRD.md)** that includes:
   - **Executive Summary**: High-level overview of the LSTM frequency filter project
   - **Problem Statement**: Clear definition of the signal processing challenge
   - **Objectives**: Measurable goals (e.g., MSE < 0.05, generalization gap < 0.01)
   - **Technical Requirements**: 
     - Data generation specifications (10,000 samples, 4 frequencies, noise parameters)
     - Model architecture requirements (LSTM, input/output dimensions)
     - Training requirements (loss function, optimizer, epochs)
   - **Functional Requirements**: 
     - Frequency extraction capability
     - One-hot selector mechanism
     - State management for L=1 sequences
   - **Non-Functional Requirements**:
     - Code quality and testing (>90% coverage)
     - Visualization requirements (14+ plots)
     - Documentation standards
   - **Success Metrics**: 
     - Training/test MSE thresholds
     - Generalization performance
     - Per-frequency accuracy
   - **Constraints**: Python 3.8-3.12, PyTorch 2.0+, computational resources

### Task 2: Architecture Design

3. **Create ARCHITECTURE.md** documenting:
   - **System Architecture**: High-level component diagram (C4 model)
   - **Data Flow**: From signal generation → training → evaluation → visualization
   - **Module Breakdown**:
     - `src/data/`: Signal generation, dataset creation, data utilities
     - `src/models/`: LSTM architecture, model factory
     - `src/training/`: Trainer, evaluator, checkpoint manager
     - `src/visualization/`: All plotting modules
     - `src/config/`: Configuration management
   - **API Design**: Key classes and their interfaces
   - **Dependencies**: External libraries and their roles

4. **Create ADR.md (Architectural Decision Records)** documenting key decisions:
   - **ADR-001**: Choice of PyTorch over TensorFlow (flexibility, ease of use)
   - **ADR-002**: L=1 sequence length with manual state management (pedagogical value)
   - **ADR-003**: MSE loss function for regression task
   - **ADR-004**: Two-layer LSTM with 64 hidden units (capacity vs. efficiency)
   - **ADR-005**: Separate train/test seeds for true generalization testing
   - **ADR-006**: Per-sample noise regeneration (realistic scenario)
   - **ADR-007**: YAML-based configuration system (flexibility)
   - **ADR-008**: Comprehensive visualization suite (14 plots for complete analysis)
   - **ADR-009**: 90%+ test coverage requirement (quality assurance)

### Task 3: Testing Strategy

5. **Create TESTING.md** documenting:
   - **Testing Philosophy**: Test-driven development approach
   - **Test Coverage Goals**: >90% code coverage
   - **Test Categories**:
     - Unit tests (signal generation, dataset, model, training components)
     - Integration tests (full pipeline execution)
     - Edge cases (empty data, single frequency, boundary conditions)
   - **Test Data Strategy**: Fixed seeds for reproducibility
   - **CI/CD Integration**: Automated testing workflow

### Task 4: Implementation Plan

6. **Create a detailed agent implementation plan** with phases:

   **Phase 1: Project Setup**
   - Initialize project structure
   - Set up configuration system
   - Create test framework
   - Document coding standards

   **Phase 2: Data Generation**
   - Implement signal generator with noise model
   - Create PyTorch Dataset and DataLoader
   - Add data persistence (save/load)
   - Write comprehensive tests

   **Phase 3: Model Architecture**
   - Implement LSTM frequency filter
   - Create model factory
   - Add state management utilities
   - Test forward/backward passes

   **Phase 4: Training Pipeline**
   - Implement trainer with state preservation
   - Add checkpoint manager
   - Create evaluator with per-frequency metrics
   - Implement validation logic

   **Phase 5: Visualization**
   - Time domain plots (signals, overlays)
   - Frequency domain plots (FFT, spectrograms)
   - Training plots (loss curves, model I/O)
   - Prediction plots (accuracy, errors)
   - Per-frequency analysis

   **Phase 6: Integration & Testing**
   - Full pipeline integration
   - End-to-end testing
   - Performance validation
   - Documentation finalization

7. **Create additional supporting documents**:
   - **CLI_USAGE_GUIDE.md**: Complete command-line interface documentation
   - **WORKFLOW_GUIDE.md**: Visual workflows and decision trees
   - **EXPERIMENTS.md**: Parameter sensitivity analysis and ablation studies
   - **EXTENSIBILITY.md**: Guide for extending the system with plugins
   - **TEST_COVERAGE_REPORT.md**: Detailed coverage analysis
   - **FINAL_QA_REPORT.md**: Comprehensive quality assessment

### Deliverables

By the end of PROMPT #2, the following documents should exist in the `docs/` folder:
- `PRD.md` - Product Requirements Document
- `ARCHITECTURE.md` - System architecture and design
- `ADR.md` - Architectural Decision Records
- `TESTING.md` - Testing strategy and requirements
- `CLI_USAGE_GUIDE.md` - Command-line interface guide
- `WORKFLOW_GUIDE.md` - Workflow diagrams and guides
- `EXPERIMENTS.md` - Experimental results and analysis
- `EXTENSIBILITY.md` - Extension and plugin guide
- `TEST_COVERAGE_REPORT.md` - Coverage analysis
- `FINAL_QA_REPORT.md` - Quality assurance report

---

## PROMPT #3: Agent-Based Implementation

Using the comprehensive documentation created in the markdown files (PRD.md, ARCHITECTURE.md, ADR.md, TESTING.md, etc.), implement the LSTM Frequency Filter project with AI agents following this workflow:

### Agent Assignment Strategy

1. **Data Agent**: Responsible for `src/data/` module
   - Read specifications from PRD.md (data requirements section)
   - Implement signal generator with per-sample noise (f'i(t) = Ai(t) · sin(2π · fi · t + φi(t)))
   - Follow noise parameters: Ai(t) ∈ (0.8, 1.2), φi(t) ∈ (0, 0.1π)
   - Create PyTorch Dataset with one-hot selector vectors
   - Implement DataLoader factory
   - Add save/load functionality for datasets
   - Write tests according to TESTING.md specifications
   - Validate against PRD success metrics

2. **Model Agent**: Responsible for `src/models/` module
   - Read architecture requirements from ARCHITECTURE.md
   - Implement 2-layer LSTM with 64 hidden units (per ADR-004)
   - Input dimension: 5 (S(t) + 4 one-hot features)
   - Output dimension: 1 (selected frequency value)
   - Implement manual state management for L=1 (per ADR-002)
   - Add state initialization and detachment methods
   - Create model factory for flexible instantiation
   - Write comprehensive model tests
   - Ensure deterministic behavior for testing

3. **Training Agent**: Responsible for `src/training/` module
   - Read training requirements from PRD.md
   - Implement LSTMTrainer with state preservation between batches
   - Use MSE loss function (per ADR-003)
   - Adam optimizer with learning rate 0.001
   - Implement gradient clipping for stability
   - Create checkpoint manager (save every 20 epochs)
   - Build evaluator with per-frequency metrics
   - Add prediction generator for visualization
   - Follow training loop specifications from ARCHITECTURE.md
   - Write training pipeline tests

4. **Visualization Agent**: Responsible for `src/visualization/` module
   - Read visualization requirements from PRD.md (14+ plots required)
   - Implement time domain plots:
     - Individual frequency signals
     - Overlay of all frequencies with mixed signal
     - Training sample structure
   - Implement frequency domain plots:
     - FFT analysis for all signals
     - Spectrogram of mixed signal
     - Complete overview combining time and frequency
   - Implement training plots:
     - Model I/O structure diagram
     - Training/validation loss curves
   - Implement prediction plots:
     - Predictions vs. actual values
     - Error distribution histograms
     - Scatter plots for accuracy
   - Implement frequency analysis plots:
     - Spectrum comparison
     - Long sequence tracking
     - Per-frequency performance metrics
   - Create base plotter class with common utilities
   - Write visualization tests

5. **Configuration Agent**: Responsible for `src/config/` module
   - Read configuration requirements from ARCHITECTURE.md
   - Implement YAML-based config loader (per ADR-007)
   - Support environment variable resolution
   - Add config validation with schema
   - Support merging default and experiment configs
   - Create config API for nested access
   - Write configuration tests

6. **Integration Agent**: Responsible for main pipeline
   - Read workflow from WORKFLOW_GUIDE.md
   - Implement `train.py` main script with CLI arguments
   - Integrate all modules: data → model → training → visualization
   - Add results summary JSON export
   - Implement full pipeline with proper error handling
   - Follow execution flow from ARCHITECTURE.md
   - Write integration tests

7. **Testing Agent**: Responsible for test suite
   - Read requirements from TESTING.md
   - Ensure >90% code coverage (per PRD requirements)
   - Write unit tests for all modules
   - Add integration tests for full pipeline
   - Test edge cases and boundary conditions
   - Ensure reproducibility with fixed seeds
   - Generate coverage reports
   - Validate all PRD success metrics are testable

8. **Documentation Agent**: Responsible for project documentation
   - Create comprehensive README.md with:
     - Project overview and motivation
     - Mathematical formulation of the problem
     - Architecture diagrams
     - Usage instructions and examples
     - Results visualization and analysis
     - Troubleshooting guide
   - Ensure all code has docstrings
   - Create examples in `examples/` folder
   - Add inline comments for complex logic
   - Generate API documentation

### Implementation Workflow

**Phase 1: Foundation (Data + Config)**
1. Configuration Agent sets up config system
2. Data Agent implements signal generation and dataset
3. Testing Agent writes data and config tests
4. Validate: All data tests pass, datasets can be generated

**Phase 2: Model Development**
1. Model Agent implements LSTM architecture
2. Testing Agent writes model tests
3. Validate: Model forward pass works, state management correct

**Phase 3: Training Pipeline**
1. Training Agent implements trainer and evaluator
2. Testing Agent writes training tests
3. Validate: Training loop executes, checkpoints save correctly

**Phase 4: Visualization**
1. Visualization Agent implements all plot types
2. Testing Agent writes visualization tests
3. Validate: All 14+ required plots generate correctly

**Phase 5: Integration**
1. Integration Agent connects all components
2. Testing Agent writes end-to-end tests
3. Validate: Full pipeline executes successfully

**Phase 6: Documentation & Polish**
1. Documentation Agent creates README and examples
2. All agents review and refine their components
3. Testing Agent validates 90%+ coverage
4. Final validation against all PRD success metrics

### Success Criteria

The implementation is complete when:
- ✅ All 125+ tests pass with >90% coverage
- ✅ Training achieves MSE < 0.05 on test set
- ✅ Generalization gap < 0.01
- ✅ All 14+ visualizations generate correctly
- ✅ Full pipeline runs end-to-end without errors
- ✅ README.md is comprehensive with examples
- ✅ Code follows Python best practices
- ✅ All PRD requirements are met
- ✅ Documentation is complete and accurate

### Agent Coordination Rules

- Each agent works on its assigned module independently
- Agents communicate through well-defined APIs (from ARCHITECTURE.md)
- Testing Agent validates each component before next phase
- Integration Agent coordinates cross-module dependencies
- All agents follow ADR decisions without deviation
- Documentation Agent ensures consistency across all docs