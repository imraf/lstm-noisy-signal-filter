"""Signal visualization tools for time/frequency analysis.

Creates visualizations for:
- Time domain signals
- Frequency domain (FFT)
- Spectrograms
- Signal overlays
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from typing import List, Tuple, Optional
from pathlib import Path


def plot_time_domain_signals(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str,
    time_window: Optional[Tuple[float, float]] = None
):
    """Plot 01: Time domain signals (pure frequencies and noisy mixed signal).
    
    Args:
        t: Time array [10000]
        frequencies: List of frequencies [f1, f2, f3, f4]
        targets: Pure target signals [4, 10000]
        S_noisy: Noisy mixed signal [10000]
        save_path: Path to save figure
        time_window: Optional (start, end) time range to plot
    """
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
    
    # Plot each pure frequency
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        axes[i].plot(t_plot, targets_plot[i], color=color, linewidth=1.5)
        axes[i].set_ylabel(f'f{i+1}={freq}Hz', fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(t_plot[0], t_plot[-1])
    
    # Plot noisy mixed signal
    axes[4].plot(t_plot, S_plot, color='black', linewidth=0.8, alpha=0.7)
    axes[4].set_ylabel('Mixed S(t)', fontsize=11, fontweight='bold')
    axes[4].set_xlabel('Time (s)', fontsize=12)
    axes[4].grid(True, alpha=0.3)
    axes[4].set_xlim(t_plot[0], t_plot[-1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_frequency_domain_fft(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str
):
    """Plot 02: Frequency domain (FFT) analysis.
    
    Args:
        t: Time array [10000]
        frequencies: List of frequencies
        targets: Pure target signals [4, 10000]
        S_noisy: Noisy mixed signal [10000]
        save_path: Path to save figure
    """
    sampling_rate = len(t) / (t[-1] - t[0])
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    fig.suptitle('Frequency Domain (FFT Analysis)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot FFT of each pure frequency
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        # Compute FFT
        fft_vals = fft(targets[i])
        fft_freq = fftfreq(len(t), 1/sampling_rate)
        
        # Only plot positive frequencies
        positive_freq_mask = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[positive_freq_mask])
        fft_freq_positive = fft_freq[positive_freq_mask]
        
        axes[i].plot(fft_freq_positive, fft_magnitude, color=color, linewidth=1.5)
        axes[i].set_ylabel(f'f{i+1}={freq}Hz', fontsize=11, fontweight='bold')
        axes[i].set_xlim(0, 15)
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot FFT of noisy mixed signal
    fft_vals = fft(S_noisy)
    fft_freq = fftfreq(len(t), 1/sampling_rate)
    positive_freq_mask = fft_freq > 0
    fft_magnitude = np.abs(fft_vals[positive_freq_mask])
    fft_freq_positive = fft_freq[positive_freq_mask]
    
    axes[4].plot(fft_freq_positive, fft_magnitude, color='black', linewidth=1.5)
    axes[4].set_ylabel('Mixed S(t)', fontsize=11, fontweight='bold')
    axes[4].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[4].set_xlim(0, 15)
    axes[4].grid(True, alpha=0.3)
    
    # Mark expected frequencies
    for freq in frequencies:
        axes[4].axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_spectrogram(
    t: np.ndarray,
    S_noisy: np.ndarray,
    frequencies: List[float],
    save_path: str
):
    """Plot 03: Spectrogram (time-frequency representation).
    
    Args:
        t: Time array [10000]
        S_noisy: Noisy mixed signal [10000]
        frequencies: List of frequencies
        save_path: Path to save figure
    """
    sampling_rate = len(t) / (t[-1] - t[0])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Compute spectrogram
    f, t_spec, Sxx = scipy_signal.spectrogram(
        S_noisy,
        fs=sampling_rate,
        nperseg=512,
        noverlap=256
    )
    
    # Plot spectrogram
    im = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title('Spectrogram of Mixed Noisy Signal', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 15)
    
    # Mark expected frequencies
    for freq in frequencies:
        ax.axhline(freq, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'{freq}Hz')
    
    plt.colorbar(im, ax=ax, label='Power (dB)')
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
    """Plot 04: Overlay of pure frequencies and mixed noisy signal.
    
    Args:
        t: Time array [10000]
        frequencies: List of frequencies
        targets: Pure target signals [4, 10000]
        S_noisy: Noisy mixed signal [10000]
        save_path: Path to save figure
        time_window: Time range to plot (default: 0-2 seconds)
    """
    mask = (t >= time_window[0]) & (t <= time_window[1])
    t_plot = t[mask]
    targets_plot = targets[:, mask]
    S_plot = S_noisy[mask]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot noisy signal as background
    ax.plot(t_plot, S_plot, color='gray', linewidth=2, alpha=0.4, label='Mixed Noisy S(t)')
    
    # Plot pure frequencies
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
    """Plot 05: Training samples structure.
    
    Shows first few samples to illustrate data format.
    
    Args:
        S: Noisy signal [40000]
        targets: Pure targets [40000]
        one_hot: One-hot vectors [40000, 4]
        frequencies: List of frequencies
        save_path: Path to save figure
        num_samples: Number of samples to display
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
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
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, num_samples + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Training Dataset Structure (First 20 Samples)', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_complete_overview(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str
):
    """Plot 00: Complete system overview with multiple visualizations.
    
    Args:
        t: Time array [10000]
        frequencies: List of frequencies
        targets: Pure target signals [4, 10000]
        S_noisy: Noisy mixed signal [10000]
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('LSTM Frequency Filter: Complete System Overview', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Top left: Time domain (zoomed)
    ax1 = fig.add_subplot(gs[0, 0])
    time_mask = t <= 2.0
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        ax1.plot(t[time_mask], targets[i, time_mask], color=color, 
                linewidth=1.5, label=f'{freq}Hz', alpha=0.7)
    ax1.plot(t[time_mask], S_noisy[time_mask], color='black', 
            linewidth=1, alpha=0.5, label='Noisy Mix')
    ax1.set_title('Time Domain (0-2s)', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Top right: Frequency domain
    ax2 = fig.add_subplot(gs[0, 1])
    sampling_rate = len(t) / (t[-1] - t[0])
    fft_vals = fft(S_noisy)
    fft_freq = fftfreq(len(t), 1/sampling_rate)
    positive_mask = fft_freq > 0
    ax2.plot(fft_freq[positive_mask], np.abs(fft_vals[positive_mask]), 
            color='black', linewidth=1.5)
    for freq in frequencies:
        ax2.axvline(freq, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.set_title('Frequency Domain (FFT)', fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_xlim(0, 15)
    ax2.grid(True, alpha=0.3)
    
    # Middle: Spectrogram
    ax3 = fig.add_subplot(gs[1, :])
    f, t_spec, Sxx = scipy_signal.spectrogram(
        S_noisy, fs=sampling_rate, nperseg=512, noverlap=256
    )
    im = ax3.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                        shading='gouraud', cmap='viridis')
    for freq in frequencies:
        ax3.axhline(freq, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax3.set_title('Spectrogram', fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_ylim(0, 15)
    plt.colorbar(im, ax=ax3, label='Power (dB)')
    
    # Bottom: Individual frequencies
    ax4 = fig.add_subplot(gs[2, :])
    time_mask = t <= 1.0
    offset = 0
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        ax4.plot(t[time_mask], targets[i, time_mask] + offset, 
                color=color, linewidth=2, label=f'{freq}Hz')
        offset += 2.5
    ax4.set_title('Individual Pure Frequencies (Stacked)', fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude (offset)')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
