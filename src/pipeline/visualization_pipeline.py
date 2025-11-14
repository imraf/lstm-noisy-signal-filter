"""Visualization pipeline execution."""

import numpy as np
from pathlib import Path

from ..data.generator import SignalGenerator
from ..visualization import signal_plots, model_plots


def generate_all_visualizations(
    train_seed: int,
    test_seed: int,
    frequencies: list,
    results: dict,
    save_dir: Path,
    verbose: bool = True
) -> None:
    """Generate all visualization plots.
    
    Args:
        train_seed: Training data seed
        test_seed: Test data seed
        frequencies: List of frequencies
        results: Training results dictionary
        save_dir: Directory to save visualizations
        verbose: Print progress
    """
    viz_dir = save_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 80)
        print("Generating Visualizations")
        print("=" * 80)
    
    gen_train = SignalGenerator(seed=train_seed)
    t = gen_train.get_time_array()
    targets_matrix_train = gen_train.generate_pure_targets()
    S_noisy_train = gen_train.generate_noisy_signal()
    
    gen_test = SignalGenerator(seed=test_seed)
    S_noisy_test = gen_test.generate_noisy_signal()
    targets_matrix_test = gen_test.generate_pure_targets()
    
    S_train, targets_train, one_hot_train = results['train_data']
    
    if verbose:
        print("[INFO] Generating signal visualizations...")
    
    signal_plots.plot_complete_overview(
        t, frequencies, targets_matrix_train, S_noisy_train,
        viz_dir / "00_complete_overview.png"
    )
    
    signal_plots.plot_time_domain_signals(
        t, frequencies, targets_matrix_train, S_noisy_train,
        viz_dir / "01_time_domain_signals.png",
        time_window=(0, 2)
    )
    
    signal_plots.plot_frequency_domain_fft(
        t, frequencies, targets_matrix_train, S_noisy_train,
        viz_dir / "02_frequency_domain_fft.png"
    )
    
    signal_plots.plot_spectrogram(
        t, S_noisy_train, frequencies,
        viz_dir / "03_spectrogram.png"
    )
    
    signal_plots.plot_overlay_signals(
        t, frequencies, targets_matrix_train, S_noisy_train,
        viz_dir / "04_overlay_signals.png"
    )
    
    signal_plots.plot_training_samples(
        S_train, targets_train, one_hot_train, frequencies,
        viz_dir / "05_training_samples.png"
    )
    
    model_plots.plot_model_io_structure(
        viz_dir / "06_model_io_structure.png"
    )
    
    if verbose:
        print("[INFO] Generating training and prediction visualizations...")
    
    model_plots.plot_training_loss(
        results['history']['train_loss'],
        results['history']['val_loss'],
        viz_dir / "07_training_loss.png"
    )
    
    samples_per_freq = 10000
    freq_idx = 1
    start_idx = freq_idx * samples_per_freq
    end_idx = start_idx + samples_per_freq
    
    test_pred_f2 = results['test_predictions'][start_idx:end_idx]
    test_target_f2 = results['test_targets'][start_idx:end_idx]
    
    model_plots.plot_predictions_vs_actual(
        t, test_pred_f2, test_target_f2, S_noisy_test,
        freq_idx, frequencies[freq_idx],
        viz_dir / "08_predictions_vs_actual.png"
    )
    
    model_plots.plot_error_distribution(
        results['test_predictions'], results['test_targets'], frequencies,
        viz_dir / "09_error_distribution.png"
    )
    
    model_plots.plot_scatter_pred_vs_actual(
        results['test_predictions'], results['test_targets'], frequencies,
        viz_dir / "10_scatter_pred_vs_actual.png"
    )
    
    model_plots.plot_frequency_spectrum_comparison(
        t, results['test_predictions'], results['test_targets'], frequencies,
        viz_dir / "11_frequency_spectrum_comparison.png"
    )
    
    model_plots.plot_long_sequence_predictions(
        t, results['test_predictions'], results['test_targets'], frequencies,
        viz_dir / "12_long_sequence_predictions.png"
    )
    
    model_plots.plot_per_frequency_metrics(
        results['test_freq_metrics'], frequencies,
        viz_dir / "13_per_frequency_metrics.png",
        split_name="Test"
    )
    
    if verbose:
        print(f"[INFO] All visualizations saved to {viz_dir}")

