"""Model visualization tools for training and prediction analysis.

Creates visualizations for:
- Training loss curves
- Prediction comparisons
- Error distributions
- Per-frequency metrics
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def plot_model_io_structure(save_path: str):
    """Plot 06: Model input/output structure diagram.
    
    Args:
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'LSTM Frequency Filter: Input/Output Structure', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Input box
    input_box = plt.Rectangle((0.5, 6), 2, 2, facecolor='#E3F2FD', 
                              edgecolor='#1976D2', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.5, 7.5, 'INPUT', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#1976D2')
    ax.text(1.5, 7.0, '[S(t), C₁, C₂, C₃, C₄]', ha='center', va='center', fontsize=10)
    ax.text(1.5, 6.5, 'Dimension: 5', ha='center', va='center', fontsize=9, style='italic')
    
    # LSTM box
    lstm_box = plt.Rectangle((3.5, 6), 3, 2, facecolor='#FFF3E0', 
                             edgecolor='#F57C00', linewidth=2)
    ax.add_patch(lstm_box)
    ax.text(5, 7.5, 'LSTM NETWORK', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#F57C00')
    ax.text(5, 7.0, 'Hidden State: (h_t, c_t)', ha='center', va='center', fontsize=10)
    ax.text(5, 6.5, 'State preserved between samples', ha='center', va='center', 
           fontsize=8, style='italic')
    
    # Output box
    output_box = plt.Rectangle((7.5, 6), 2, 2, facecolor='#E8F5E9', 
                               edgecolor='#388E3C', linewidth=2)
    ax.add_patch(output_box)
    ax.text(8.5, 7.5, 'OUTPUT', ha='center', va='center', 
           fontsize=14, fontweight='bold', color='#388E3C')
    ax.text(8.5, 7.0, 'Pure Frequency', ha='center', va='center', fontsize=10)
    ax.text(8.5, 6.5, 'Dimension: 1', ha='center', va='center', fontsize=9, style='italic')
    
    # Arrows
    ax.annotate('', xy=(3.5, 7), xytext=(2.5, 7), 
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(7.5, 7), xytext=(6.5, 7), 
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Details boxes
    detail1 = plt.Rectangle((0.5, 3.5), 4, 2, facecolor='#F5F5F5', 
                           edgecolor='gray', linewidth=1, linestyle='--')
    ax.add_patch(detail1)
    ax.text(2.5, 5.2, 'Input Components:', ha='center', va='top', 
           fontsize=11, fontweight='bold')
    ax.text(2.5, 4.8, '• S(t): Mixed noisy signal sample', ha='center', va='top', fontsize=9)
    ax.text(2.5, 4.5, '• C: One-hot selection vector', ha='center', va='top', fontsize=9)
    ax.text(2.5, 4.2, '  [1,0,0,0] → Extract f₁=1Hz', ha='center', va='top', fontsize=9)
    ax.text(2.5, 3.9, '  [0,1,0,0] → Extract f₂=3Hz', ha='center', va='top', fontsize=9)
    
    detail2 = plt.Rectangle((5.5, 3.5), 4, 2, facecolor='#F5F5F5', 
                           edgecolor='gray', linewidth=1, linestyle='--')
    ax.add_patch(detail2)
    ax.text(7.5, 5.2, 'Processing:', ha='center', va='top', 
           fontsize=11, fontweight='bold')
    ax.text(7.5, 4.8, '• Conditional regression', ha='center', va='top', fontsize=9)
    ax.text(7.5, 4.5, '• L=1: Sequence length of 1', ha='center', va='top', fontsize=9)
    ax.text(7.5, 4.2, '• State management critical', ha='center', va='top', fontsize=9)
    ax.text(7.5, 3.9, '• Learns frequency structure', ha='center', va='top', fontsize=9)
    
    # Example flow
    example_box = plt.Rectangle((1, 0.5), 8, 2.5, facecolor='#FFF9C4', 
                               edgecolor='#F9A825', linewidth=2)
    ax.add_patch(example_box)
    ax.text(5, 2.7, 'Example Data Flow', ha='center', va='center', 
           fontsize=12, fontweight='bold', color='#F9A825')
    ax.text(5, 2.3, 'Sample at t=0.000s, extracting f₂=3Hz:', ha='center', va='center', fontsize=10)
    ax.text(5, 1.9, '[0.8124, 0, 1, 0, 0] → LSTM → 0.0000', ha='center', va='center', 
           fontsize=10, family='monospace')
    ax.text(5, 1.5, 'Sample at t=0.001s, extracting f₂=3Hz:', ha='center', va='center', fontsize=10)
    ax.text(5, 1.1, '[0.7932, 0, 1, 0, 0] → LSTM → 0.0188', ha='center', va='center', 
           fontsize=10, family='monospace')
    ax.text(5, 0.7, '(Hidden state preserved between consecutive samples)', 
           ha='center', va='center', fontsize=8, style='italic', color='#F57C00')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]],
    save_path: str
):
    """Plot 07: Training loss curve.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    
    if val_losses and len(val_losses) > 0:
        ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Training Progress: Loss vs Epoch', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add final loss value annotation
    final_train_loss = train_losses[-1]
    ax.annotate(f'Final: {final_train_loss:.6f}', 
               xy=(len(train_losses), final_train_loss),
               xytext=(len(train_losses) * 0.7, final_train_loss * 1.2),
               arrowprops=dict(arrowstyle='->', color='blue'),
               fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


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
    """Plot 08: Predictions vs actual for single frequency.
    
    Shows Target (line), LSTM output (dots), and noisy S (chaotic background).
    
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
    
    # Plot noisy background
    ax.plot(t_plot, S_plot, color='gray', linewidth=1, alpha=0.3, 
           label='Mixed Noisy S(t) (background)', zorder=1)
    
    # Plot target (ground truth)
    ax.plot(t_plot, target_plot, 'g-', linewidth=2.5, 
           label=f'Target (Pure f{freq_idx+1}={frequency}Hz)', zorder=3, alpha=0.8)
    
    # Plot predictions as dots
    ax.scatter(t_plot, pred_plot, c='red', s=10, alpha=0.7, 
              label='LSTM Predictions', zorder=4)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title(f'Prediction Quality: Extracting f{freq_idx+1}={frequency}Hz from Noisy Signal', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Compute and display MSE
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
    """Plot 09: Error distribution for all frequencies.
    
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
        
        # Statistics
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
    """Plot 10: Scatter plot of predictions vs actual.
    
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
        
        # Perfect prediction line
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
        
        # R² score
        correlation = np.corrcoef(freq_targets, freq_preds)[0, 1]
        r_squared = correlation ** 2
        axes[i].text(0.02, 0.98, f'R²: {r_squared:.4f}', 
                    transform=axes[i].transAxes, fontsize=10, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_frequency_spectrum_comparison(
    t: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot 11: Frequency spectrum comparison.
    
    Args:
        t: Time array [10000]
        predictions: All predictions [40000]
        targets: All targets [40000]
        frequencies: List of frequencies
        save_path: Path to save figure
    """
    from scipy.fft import fft, fftfreq
    
    sampling_rate = len(t) / (t[-1] - t[0])
    samples_per_freq = len(predictions) // len(frequencies)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle('Frequency Spectrum: Predictions vs Targets', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx = i * samples_per_freq
        end_idx = start_idx + samples_per_freq
        freq_preds = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        # Compute FFT
        fft_pred = fft(freq_preds)
        fft_target = fft(freq_targets)
        fft_freq = fftfreq(len(freq_preds), 1/sampling_rate)
        
        # Only positive frequencies
        positive_mask = fft_freq > 0
        fft_freq_pos = fft_freq[positive_mask]
        fft_pred_mag = np.abs(fft_pred[positive_mask])
        fft_target_mag = np.abs(fft_target[positive_mask])
        
        axes[i].plot(fft_freq_pos, fft_target_mag, color='green', 
                    linewidth=2, label='Target', alpha=0.7)
        axes[i].plot(fft_freq_pos, fft_pred_mag, color=color, 
                    linewidth=2, label='Prediction', alpha=0.7, linestyle='--')
        axes[i].axvline(freq, color='red', linestyle=':', linewidth=2, 
                       alpha=0.5, label=f'{freq}Hz')
        
        axes[i].set_xlabel('Frequency (Hz)', fontsize=11)
        axes[i].set_ylabel('Magnitude', fontsize=11)
        axes[i].set_title(f'f{i+1}={freq}Hz', fontsize=12, fontweight='bold')
        axes[i].set_xlim(0, 15)
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_long_sequence_predictions(
    t: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot 12: Long sequence predictions for all frequencies.
    
    Args:
        t: Time array [10000]
        predictions: All predictions [40000]
        targets: All targets [40000]
        frequencies: List of frequencies
        save_path: Path to save figure
    """
    samples_per_freq = len(predictions) // len(frequencies)
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    fig.suptitle('Long Sequence Predictions (Full 10 seconds)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        start_idx = i * samples_per_freq
        end_idx = start_idx + samples_per_freq
        freq_preds = predictions[start_idx:end_idx]
        freq_targets = targets[start_idx:end_idx]
        
        axes[i].plot(t, freq_targets, color='green', linewidth=1.5, 
                    label='Target', alpha=0.6)
        axes[i].plot(t, freq_preds, color=color, linewidth=1, 
                    label='Prediction', alpha=0.8)
        
        axes[i].set_ylabel(f'f{i+1}={freq}Hz', fontsize=11, fontweight='bold')
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 10)
        
        # MSE annotation
        mse = np.mean((freq_preds - freq_targets) ** 2)
        axes[i].text(0.02, 0.98, f'MSE: {mse:.6f}', transform=axes[i].transAxes, 
                    fontsize=9, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_frequency_metrics(
    metrics: Dict[int, Dict[str, float]],
    frequencies: List[float],
    save_path: str,
    split_name: str = 'Test'
):
    """Plot 13: Per-frequency performance metrics.
    
    Args:
        metrics: Dictionary mapping freq_idx to metrics
        frequencies: List of frequencies
        save_path: Path to save figure
        split_name: 'Train' or 'Test'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Per-Frequency Performance Metrics ({split_name} Set)', 
                fontsize=16, fontweight='bold')
    
    freq_labels = [f'f{i+1}={freq}Hz' for i, freq in enumerate(frequencies)]
    mse_values = [metrics[i]['mse'] for i in range(len(frequencies))]
    mae_values = [metrics[i]['mae'] for i in range(len(frequencies))]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # MSE bar chart
    bars1 = ax1.bar(freq_labels, mse_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Mean Squared Error', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, mse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    # MAE bar chart
    bars2 = ax2.bar(freq_labels, mae_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars2, mae_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
