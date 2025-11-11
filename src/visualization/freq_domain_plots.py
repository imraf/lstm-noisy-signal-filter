"""Frequency domain signal visualization tools."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
from typing import List


def plot_frequency_domain_fft(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str
):
    """Plot FFT analysis."""
    sampling_rate = len(t) / (t[-1] - t[0])
    
    fig, axes = plt.subplots(5, 1, figsize=(14, 10))
    fig.suptitle('Frequency Domain (FFT Analysis)', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        fft_vals = fft(targets[i])
        fft_freq = fftfreq(len(t), 1/sampling_rate)
        positive_freq_mask = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[positive_freq_mask])
        fft_freq_positive = fft_freq[positive_freq_mask]
        
        axes[i].plot(fft_freq_positive, fft_magnitude, color=color, linewidth=1.5)
        axes[i].set_ylabel(f'f{i+1}={freq}Hz', fontsize=11, fontweight='bold')
        axes[i].set_xlim(0, 15)
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
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
    """Plot spectrogram."""
    sampling_rate = len(t) / (t[-1] - t[0])
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    f, t_spec, Sxx = scipy_signal.spectrogram(
        S_noisy, fs=sampling_rate, nperseg=512, noverlap=256
    )
    
    im = ax.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title('Spectrogram of Mixed Noisy Signal', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 15)
    
    for freq in frequencies:
        ax.axhline(freq, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'{freq}Hz')
    
    plt.colorbar(im, ax=ax, label='Power (dB)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_complete_overview(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str
):
    """Plot complete system overview."""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle('LSTM Frequency Filter: Complete System Overview', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
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

