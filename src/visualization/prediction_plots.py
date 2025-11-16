"""Prediction visualization tools."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from .plot_utils import setup_figure, save_and_close, get_default_colors, format_axis, add_metric_text


def plot_predictions_vs_actual(
    t: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    S_noisy: np.ndarray,
    freq_idx: int,
    frequency: float,
    save_path: str,
    time_window: Tuple[float, float] = (0, 2)
):
    """Plot predictions vs actual for single frequency."""
    mask = (t >= time_window[0]) & (t <= time_window[1])
    t_plot, pred_plot, target_plot, S_plot = t[mask], predictions[mask], targets[mask], S_noisy[mask]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(t_plot, S_plot, color='gray', linewidth=1, alpha=0.3, 
           label='Mixed Noisy S(t)', zorder=1)
    ax.plot(t_plot, target_plot, 'g-', linewidth=2.5, 
           label=f'Target (Pure f{freq_idx+1}={frequency}Hz)', zorder=3, alpha=0.8)
    ax.scatter(t_plot, pred_plot, c='red', s=10, alpha=0.7, 
              label='LSTM Predictions', zorder=4)
    
    format_axis(ax, 'Time (s)', 'Amplitude', 
                f'Prediction Quality: Extracting f{freq_idx+1}={frequency}Hz from Noisy Signal')
    ax.legend(loc='upper right', fontsize=11)
    
    mse = np.mean((pred_plot - target_plot) ** 2)
    add_metric_text(ax, f'MSE: {mse:.6f}')
    
    save_and_close(save_path)


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot error distribution for all frequencies."""
    errors = predictions - targets
    samples_per_freq = len(predictions) // len(frequencies)
    
    fig, axes = setup_figure(2, 2, (14, 10), 'Error Distribution per Frequency')
    axes = axes.flatten()
    colors = get_default_colors()
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx, end_idx = i * samples_per_freq, (i + 1) * samples_per_freq
        freq_errors = errors[start_idx:end_idx]
        
        axes[i].hist(freq_errors, bins=50, color=color, alpha=0.7, edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
        format_axis(axes[i], 'Prediction Error', 'Frequency Count', f'f{i+1}={freq}Hz')
        
        add_metric_text(axes[i], f'Mean: {np.mean(freq_errors):.4f}\nStd: {np.std(freq_errors):.4f}')
    
    save_and_close(save_path)


def plot_scatter_pred_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot scatter plot of predictions vs actual."""
    samples_per_freq = len(predictions) // len(frequencies)
    
    fig, axes = setup_figure(2, 2, (14, 10), 'Predictions vs Actual Values')
    axes = axes.flatten()
    colors = get_default_colors()
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx, end_idx = i * samples_per_freq, (i + 1) * samples_per_freq
        freq_preds, freq_targets = predictions[start_idx:end_idx], targets[start_idx:end_idx]
        
        axes[i].scatter(freq_targets, freq_preds, c=color, alpha=0.3, s=1)
        
        min_val, max_val = min(freq_targets.min(), freq_preds.min()), max(freq_targets.max(), freq_preds.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
        
        format_axis(axes[i], 'Actual Values', 'Predicted Values', f'f{i+1}={freq}Hz')
        axes[i].legend(loc='upper left', fontsize=9)
        axes[i].set_aspect('equal', adjustable='box')
        
        r_squared = np.corrcoef(freq_targets, freq_preds)[0, 1] ** 2
        add_metric_text(axes[i], f'R²: {r_squared:.4f}')
    
    save_and_close(save_path)

