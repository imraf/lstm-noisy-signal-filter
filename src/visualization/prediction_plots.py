"""Prediction visualization tools.

Creates visualizations for:
- Predictions vs actual comparisons
- Error distributions
- Scatter plots
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple


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
    """Plot predictions vs actual for single frequency.
    
    Args:
        t: Time array [10000]
        predictions: Model predictions for this frequency [10000]
        targets: Ground truth targets [10000]
        S_noisy: Noisy mixed signal [10000]
        freq_idx: Frequency index
        frequency: Frequency value (Hz)
        save_path: Path to save figure
        time_window: Time range to plot
    """
    mask = (t >= time_window[0]) & (t <= time_window[1])
    t_plot = t[mask]
    pred_plot = predictions[mask]
    target_plot = targets[mask]
    S_plot = S_noisy[mask]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(t_plot, S_plot, color='gray', linewidth=1, alpha=0.3, 
           label='Mixed Noisy S(t) (background)', zorder=1)
    
    ax.plot(t_plot, target_plot, 'g-', linewidth=2.5, 
           label=f'Target (Pure f{freq_idx+1}={frequency}Hz)', zorder=3, alpha=0.8)
    
    ax.scatter(t_plot, pred_plot, c='red', s=10, alpha=0.7, 
              label='LSTM Predictions', zorder=4)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Prediction Quality: Extracting f{freq_idx+1}={frequency}Hz from Noisy Signal', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    mse = np.mean((pred_plot - target_plot) ** 2)
    ax.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=ax.transAxes, 
           fontsize=12, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot error distribution for all frequencies.
    
    Args:
        predictions: All predictions [40000]
        targets: All targets [40000]
        frequencies: List of frequencies
        save_path: Path to save figure
    """
    errors = predictions - targets
    samples_per_freq = len(predictions) // len(frequencies)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Error Distribution per Frequency', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx = i * samples_per_freq
        end_idx = start_idx + samples_per_freq
        freq_errors = errors[start_idx:end_idx]
        
        axes[i].hist(freq_errors, bins=50, color=color, alpha=0.7, edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[i].set_xlabel('Prediction Error', fontsize=11)
        axes[i].set_ylabel('Frequency Count', fontsize=11)
        axes[i].set_title(f'f{i+1}={freq}Hz', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        mean_error = np.mean(freq_errors)
        std_error = np.std(freq_errors)
        axes[i].text(0.02, 0.98, f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
                    transform=axes[i].transAxes, fontsize=10, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_pred_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot scatter plot of predictions vs actual.
    
    Args:
        predictions: All predictions [40000]
        targets: All targets [40000]
        frequencies: List of frequencies
        save_path: Path to save figure
    """
    samples_per_freq = len(predictions) // len(frequencies)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Predictions vs Actual Values', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx = i * samples_per_freq
        end_idx = start_idx + samples_per_freq
        freq_preds = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        axes[i].scatter(freq_targets, freq_preds, c=color, alpha=0.3, s=1)
        
        min_val = min(freq_targets.min(), freq_preds.min())
        max_val = max(freq_targets.max(), freq_preds.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='Perfect Prediction')
        
        axes[i].set_xlabel('Actual Values', fontsize=11)
        axes[i].set_ylabel('Predicted Values', fontsize=11)
        axes[i].set_title(f'f{i+1}={freq}Hz', fontsize=12, fontweight='bold')
        axes[i].legend(loc='upper left', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
        
        correlation = np.corrcoef(freq_targets, freq_preds)[0, 1]
        r_squared = correlation ** 2
        axes[i].text(0.02, 0.98, f'R²: {r_squared:.4f}', 
                    transform=axes[i].transAxes, fontsize=10, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

