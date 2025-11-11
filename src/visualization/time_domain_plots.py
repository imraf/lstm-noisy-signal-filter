"""Time domain signal visualization tools."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional


def plot_time_domain_signals(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str,
    time_window: Optional[Tuple[float, float]] = None
):
    """Plot time domain signals."""
    if time_window:
        mask = (t >= time_window[0]) & (t <= time_window[1])
        t_plot = t[mask]
        targets_plot = targets[:, mask]
        S_plot = S_noisy[mask]
    else:
        t_plot = t
        targets_plot = targets
        S_plot = S_noisy
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    fig.suptitle('Time Domain Signals', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        axes[i].plot(t_plot, targets_plot[i], color=color, linewidth=1.5)
        axes[i].set_ylabel(f'f{i+1}={freq}Hz', fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(t_plot[0], t_plot[-1])
    
    axes[4].plot(t_plot, S_plot, color='black', linewidth=0.8, alpha=0.7)
    axes[4].set_ylabel('Mixed S(t)', fontsize=11, fontweight='bold')
    axes[4].set_xlabel('Time (s)', fontsize=12)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlim(t_plot[0], t_plot[-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_overlay_signals(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str,
    time_window: Tuple[float, float] = (0, 2)
):
    """Plot overlay of pure frequencies and mixed signal."""
    mask = (t >= time_window[0]) & (t <= time_window[1])
    t_plot = t[mask]
    targets_plot = targets[:, mask]
    S_plot = S_noisy[mask]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t_plot, S_plot, color='gray', linewidth=2, alpha=0.4, label='Mixed Noisy S(t)')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        ax.plot(t_plot, targets_plot[i], color=color, linewidth=2, 
                label=f'f{i+1}={freq}Hz (pure)', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Overlay: Pure Frequencies vs Mixed Noisy Signal', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_samples(
    S: np.ndarray,
    targets: np.ndarray,
    one_hot: np.ndarray,
    frequencies: List[float],
    save_path: str,
    num_samples: int = 20
):
    """Plot training samples structure."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    headers = ['Index', 'Time', 'S(t)', 'C (One-Hot)', 'Target', 'Frequency']
    
    for i in range(num_samples):
        freq_idx = np.argmax(one_hot[i])
        one_hot_str = ', '.join([f'{val:.0f}' for val in one_hot[i]])
        table_data.append([
            f'{i}',
            f'{i // 4 * 0.001:.3f}',
            f'{S[i]:.4f}',
            f'[{one_hot_str}]',
            f'{targets[i]:.4f}',
            f'{frequencies[freq_idx]}Hz'
        ])
    
    table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, num_samples + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Training Dataset Structure (First 20 Samples)', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

