"""Main training script for LSTM Frequency Filter.

Orchestrates:
1. Dataset generation
2. Model training
3. Evaluation
4. Visualization generation
"""

import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.data.generator import SignalGenerator, create_train_test_datasets
from src.data.dataset import create_dataloaders, save_dataset
from src.models.lstm_filter import create_model
from src.training.trainer import LSTMTrainer
from src.training.evaluator import ModelEvaluator, evaluate_model
from src.visualization import signal_plots
from src.visualization import model_plots


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("LSTM Frequency Filter - Training Pipeline")
    print("=" * 80)
    
    # Configuration
    TRAIN_SEED = 11
    TEST_SEED = 42
    FREQUENCIES = [1.0, 3.0, 5.0, 7.0]
    
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    
    # Setup paths
    output_dir = Path("outputs")
    viz_dir = output_dir / "visualizations"
    model_dir = output_dir / "models"
    dataset_dir = output_dir / "datasets"
    
    for dir_path in [viz_dir, model_dir, dataset_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Using device: {device}")
    
    # ========================================================================
    # STEP 1: Generate Datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Generating Datasets")
    print("=" * 80)
    
    print(f"[INFO] Generating training data with seed {TRAIN_SEED}...")
    print(f"[INFO] Generating test data with seed {TEST_SEED}...")
    
    train_data, test_data = create_train_test_datasets(
        train_seed=TRAIN_SEED,
        test_seed=TEST_SEED
    )
    
    S_train, targets_train, one_hot_train = train_data
    S_test, targets_test, one_hot_test = test_data
    
    print(f"[INFO] Training set: {len(targets_train)} samples")
    print(f"[INFO] Test set: {len(targets_test)} samples")
    
    # Save datasets
    save_dataset(S_train, targets_train, one_hot_train, 
                dataset_dir / f"train_data_seed{TRAIN_SEED}.pt")
    save_dataset(S_test, targets_test, one_hot_test, 
                dataset_dir / f"test_data_seed{TEST_SEED}.pt")
    print(f"[INFO] Datasets saved to {dataset_dir}")
    
    # Get time array and targets for visualization
    gen = SignalGenerator(seed=TRAIN_SEED)
    t = gen.get_time_array()
    targets_matrix_train = gen.generate_pure_targets()
    S_noisy_train = gen.generate_noisy_signal()
    
    gen_test = SignalGenerator(seed=TEST_SEED)
    S_noisy_test = gen_test.generate_noisy_signal()
    targets_matrix_test = gen_test.generate_pure_targets()
    
    # ========================================================================
    # STEP 2: Create DataLoaders
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Creating DataLoaders")
    print("=" * 80)
    
    train_loader, test_loader = create_dataloaders(
        S_train, targets_train, one_hot_train,
        S_test, targets_test, one_hot_test,
        batch_size=BATCH_SIZE,
        shuffle_train=False  # CRITICAL: False for L=1 sequential training
    )
    
    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Test batches: {len(test_loader)}")
    print(f"[INFO] Batch size: {BATCH_SIZE}")
    
    # ========================================================================
    # STEP 3: Generate Signal Visualizations (Before Training)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Generating Signal Visualizations")
    print("=" * 80)
    
    print("[INFO] Generating 00_complete_overview...")
    signal_plots.plot_complete_overview(
        t, FREQUENCIES, targets_matrix_train, S_noisy_train,
        viz_dir / "00_complete_overview.png"
    )
    
    print("[INFO] Generating 01_time_domain_signals...")
    signal_plots.plot_time_domain_signals(
        t, FREQUENCIES, targets_matrix_train, S_noisy_train,
        viz_dir / "01_time_domain_signals.png",
        time_window=(0, 2)
    )
    
    print("[INFO] Generating 02_frequency_domain_fft...")
    signal_plots.plot_frequency_domain_fft(
        t, FREQUENCIES, targets_matrix_train, S_noisy_train,
        viz_dir / "02_frequency_domain_fft.png"
    )
    
    print("[INFO] Generating 03_spectrogram...")
    signal_plots.plot_spectrogram(
        t, S_noisy_train, FREQUENCIES,
        viz_dir / "03_spectrogram.png"
    )
    
    print("[INFO] Generating 04_overlay_signals...")
    signal_plots.plot_overlay_signals(
        t, FREQUENCIES, targets_matrix_train, S_noisy_train,
        viz_dir / "04_overlay_signals.png"
    )
    
    print("[INFO] Generating 05_training_samples...")
    signal_plots.plot_training_samples(
        S_train, targets_train, one_hot_train, FREQUENCIES,
        viz_dir / "05_training_samples.png"
    )
    
    print("[INFO] Generating 06_model_io_structure...")
    model_plots.plot_model_io_structure(
        viz_dir / "06_model_io_structure.png"
    )
    
    # ========================================================================
    # STEP 4: Create and Train Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Training LSTM Model")
    print("=" * 80)
    
    model = create_model(
        input_size=5,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=1,
        dropout=0.2,
        device=device
    )
    
    print(f"[INFO] Model architecture:")
    print(f"       - Input size: 5 [S(t), C1, C2, C3, C4]")
    print(f"       - Hidden size: {HIDDEN_SIZE}")
    print(f"       - Num layers: {NUM_LAYERS}")
    print(f"       - Output size: 1")
    print(f"       - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = LSTMTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE
    )
    
    print(f"\n[INFO] Starting training for {NUM_EPOCHS} epochs...")
    print(f"[INFO] Learning rate: {LEARNING_RATE}")
    print("-" * 80)
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        save_path=str(model_dir),
        save_every=20,
        verbose=True
    )
    
    print("-" * 80)
    print("[INFO] Training completed!")
    
    # Save final model
    final_model_path = model_dir / f"lstm_l1_epoch{NUM_EPOCHS}_final.pth"
    trainer.save_checkpoint(final_model_path, NUM_EPOCHS)
    print(f"[INFO] Final model saved to {final_model_path}")
    
    # Plot training loss
    print("\n[INFO] Generating 07_training_loss...")
    model_plots.plot_training_loss(
        history['train_loss'],
        history['val_loss'],
        viz_dir / "07_training_loss.png"
    )
    
    # ========================================================================
    # STEP 5: Evaluate Model
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Evaluating Model")
    print("=" * 80)
    
    evaluator = ModelEvaluator(model, device)
    
    # Overall metrics
    print("[INFO] Computing overall metrics...")
    train_mse = evaluator.compute_mse(train_loader)
    test_mse = evaluator.compute_mse(test_loader)
    
    print(f"\n[RESULTS] Train MSE: {train_mse:.8f}")
    print(f"[RESULTS] Test MSE:  {test_mse:.8f}")
    print(f"[RESULTS] Generalization Gap: {abs(test_mse - train_mse):.8f}")
    
    if abs(test_mse - train_mse) < 0.01:
        print("[SUCCESS] Model generalizes well! (Gap < 0.01)")
    else:
        print("[WARNING] Model may be overfitting or underfitting")
    
    # Per-frequency metrics
    print("\n[INFO] Computing per-frequency metrics...")
    train_freq_metrics = evaluator.evaluate_per_frequency(train_loader)
    test_freq_metrics = evaluator.evaluate_per_frequency(test_loader)
    
    print("\nPer-Frequency Performance (Test Set):")
    for i, freq in enumerate(FREQUENCIES):
        metrics = test_freq_metrics[i]
        print(f"  f{i+1}={freq}Hz: MSE={metrics['mse']:.8f}, MAE={metrics['mae']:.8f}")
    
    # Generate predictions
    print("\n[INFO] Generating predictions for visualization...")
    train_predictions, train_targets = evaluator.generate_predictions(train_loader)
    test_predictions, test_targets = evaluator.generate_predictions(test_loader)
    
    # ========================================================================
    # STEP 6: Generate Model Performance Visualizations
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Generating Model Performance Visualizations")
    print("=" * 80)
    
    # Extract predictions for frequency f2 (index 1)
    samples_per_freq = 10000
    freq_idx = 1  # f2 = 3Hz
    start_idx = freq_idx * samples_per_freq
    end_idx = start_idx + samples_per_freq
    
    test_pred_f2 = test_predictions[start_idx:end_idx]
    test_target_f2 = test_targets[start_idx:end_idx]
    
    # Extract just the samples for this frequency from expanded S
    S_test_samples = S_test[start_idx:end_idx]
    
    print("[INFO] Generating 08_predictions_vs_actual...")
    model_plots.plot_predictions_vs_actual(
        t, test_pred_f2, test_target_f2, S_noisy_test,
        freq_idx, FREQUENCIES[freq_idx],
        viz_dir / "08_predictions_vs_actual.png"
    )
    
    print("[INFO] Generating 09_error_distribution...")
    model_plots.plot_error_distribution(
        test_predictions, test_targets, FREQUENCIES,
        viz_dir / "09_error_distribution.png"
    )
    
    print("[INFO] Generating 10_scatter_pred_vs_actual...")
    model_plots.plot_scatter_pred_vs_actual(
        test_predictions, test_targets, FREQUENCIES,
        viz_dir / "10_scatter_pred_vs_actual.png"
    )
    
    print("[INFO] Generating 11_frequency_spectrum_comparison...")
    model_plots.plot_frequency_spectrum_comparison(
        t, test_predictions, test_targets, FREQUENCIES,
        viz_dir / "11_frequency_spectrum_comparison.png"
    )
    
    print("[INFO] Generating 12_long_sequence_predictions...")
    model_plots.plot_long_sequence_predictions(
        t, test_predictions, test_targets, FREQUENCIES,
        viz_dir / "12_long_sequence_predictions.png"
    )
    
    print("[INFO] Generating 13_per_frequency_metrics...")
    model_plots.plot_per_frequency_metrics(
        test_freq_metrics, FREQUENCIES,
        viz_dir / "13_per_frequency_metrics.png",
        split_name="Test"
    )
    
    # ========================================================================
    # STEP 7: Save Results Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: Saving Results Summary")
    print("=" * 80)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'train_seed': TRAIN_SEED,
            'test_seed': TEST_SEED,
            'frequencies': FREQUENCIES,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'num_epochs': NUM_EPOCHS
        },
        'metrics': {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'generalization_gap': float(abs(test_mse - train_mse)),
            'generalizes_well': abs(test_mse - train_mse) < 0.01
        },
        'per_frequency': {
            'train': {f'f{i+1}_{freq}Hz': train_freq_metrics[i] 
                     for i, freq in enumerate(FREQUENCIES)},
            'test': {f'f{i+1}_{freq}Hz': test_freq_metrics[i] 
                    for i, freq in enumerate(FREQUENCIES)}
        }
    }
    
    results_file = output_dir / "results_summary.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[INFO] Results summary saved to {results_file}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\n[SUMMARY]")
    print(f"  - Trained for {NUM_EPOCHS} epochs")
    print(f"  - Final Train MSE: {train_mse:.8f}")
    print(f"  - Final Test MSE: {test_mse:.8f}")
    print(f"  - Generated 14 visualization plots")
    print(f"  - Model saved to: {model_dir}")
    print(f"  - Visualizations saved to: {viz_dir}")
    print(f"  - Results saved to: {results_file}")
    print(f"\n[NEXT STEPS]")
    print(f"  1. Review visualizations in {viz_dir}")
    print(f"  2. Check results_summary.json for detailed metrics")
    print(f"  3. Run tests: pytest tests/ --cov=src")
    print(f"  4. Generate README.md with visualizations")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
