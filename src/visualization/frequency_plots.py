"""Frequency analysis visualization tools."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from .plot_utils import (
    setup_figure, compute_fft, get_positive_spectrum, 
    save_and_close, get_default_colors, format_axis
)


def plot_frequency_spectrum_comparison(
    t: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot frequency spectrum comparison."""
    sampling_rate = len(t) / (t[-1] - t[0])
    samples_per_freq = len(predictions) // len(frequencies)
    fig, axes = setup_figure(2, 2, (14, 10), 'Frequency Spectrum: Predictions vs Targets')
    axes = axes.flatten()
    colors = get_default_colors()
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx = i * samples_per_freq
        end_idx = start_idx + samples_per_freq
        freq_preds = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        fft_pred, fft_freq = compute_fft(freq_preds, sampling_rate)
        fft_target, _ = compute_fft(freq_targets, sampling_rate)
        fft_pred_mag, fft_freq_pos = get_positive_spectrum(fft_pred, fft_freq)
        fft_target_mag, _ = get_positive_spectrum(fft_target, fft_freq)
        
        axes[i].plot(fft_freq_pos, fft_target_mag, color='green', linewidth=2, label='Target', alpha=0.7)
        axes[i].plot(fft_freq_pos, fft_pred_mag, color=color, linewidth=2, label='Prediction', alpha=0.7, linestyle='--')
        axes[i].axvline(freq, color='red', linestyle=':', linewidth=2, alpha=0.5, label=f'{freq}Hz')
        format_axis(axes[i], 'Frequency (Hz)', 'Magnitude', f'f{i+1}={freq}Hz')
        axes[i].set_xlim(0, 15)
        axes[i].legend(loc='upper right', fontsize=9)
    
    save_and_close(save_path)


def plot_long_sequence_predictions(
    t: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot long sequence predictions for all frequencies."""
    samples_per_freq = len(predictions) // len(frequencies)
    fig, axes = setup_figure(4, 1, (16, 12), 'Long Sequence Predictions (Full 10 seconds)')
    colors = get_default_colors()
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx = i * samples_per_freq
        end_idx = start_idx + samples_per_freq
        freq_preds = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        axes[i].plot(t, freq_targets, color='green', linewidth=1.5, label='Target', alpha=0.6)
        axes[i].plot(t, freq_preds, color=color, linewidth=1, label='Prediction', alpha=0.8)
        format_axis(axes[i], ylabel=f'f{i+1}={freq}Hz')
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].set_xlim(0, 10)
        
        mse = np.mean((freq_preds - freq_targets) ** 2)
        axes[i].text(0.02, 0.98, f'MSE: {mse:.6f}', transform=axes[i].transAxes, 
                    fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    save_and_close(save_path)


def plot_per_frequency_metrics(
    metrics: Dict[int, Dict[str, float]],
    frequencies: List[float],
    save_path: str,
    split_name: str = 'Test'
):
    """Plot per-frequency performance metrics."""
    fig, (ax1, ax2) = setup_figure(1, 2, (14, 6), f'Per-Frequency Performance Metrics ({split_name} Set)')
    freq_labels = [f'f{i+1}={freq}Hz' for i, freq in enumerate(frequencies)]
    mse_values = [metrics[i]['mse'] for i in range(len(frequencies))]
    mae_values = [metrics[i]['mae'] for i in range(len(frequencies))]
    colors = get_default_colors()
    
    bars1 = ax1.bar(freq_labels, mse_values, color=colors, alpha=0.7, edgecolor='black')
    format_axis(ax1, ylabel='MSE', title='Mean Squared Error')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, mse_values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    bars2 = ax2.bar(freq_labels, mae_values, color=colors, alpha=0.7, edgecolor='black')
    format_axis(ax2, ylabel='MAE', title='Mean Absolute Error')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, mae_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    save_and_close(save_path)

