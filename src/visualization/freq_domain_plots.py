"""Frequency domain signal visualization tools."""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import signal as scipy_signal
from typing import List
from .plot_utils import (
    setup_figure, save_and_close, get_default_colors,
    plot_fft_spectrum, add_frequency_markers, format_axis
)


def plot_frequency_domain_fft(
    t: np.ndarray,
    frequencies: List[float],
    targets: np.ndarray,
    S_noisy: np.ndarray,
    save_path: str
):
    """Plot FFT analysis."""
    sampling_rate = len(t) / (t[-1] - t[0])
    fig, axes = setup_figure(5, 1, (14, 10), 'Frequency Domain (FFT Analysis)')
    colors = get_default_colors()
    
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        plot_fft_spectrum(axes[i], targets[i], sampling_rate, color=color)
        format_axis(axes[i], ylabel=f'f{i+1}={freq}Hz')
        add_frequency_markers(axes[i], [freq])
    
    plot_fft_spectrum(axes[4], S_noisy, sampling_rate, color='black')
    format_axis(axes[4], xlabel='Frequency (Hz)', ylabel='Mixed S(t)')
    add_frequency_markers(axes[4], frequencies)
    
    save_and_close(save_path)


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
    format_axis(ax, xlabel='Time (s)', ylabel='Frequency (Hz)', 
                title='Spectrogram of Mixed Noisy Signal')
    ax.set_ylim(0, 15)
    
    for freq in frequencies:
        ax.axhline(freq, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.colorbar(im, ax=ax, label='Power (dB)')
    save_and_close(save_path)


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
    
    colors = get_default_colors()
    sampling_rate = len(t) / (t[-1] - t[0])
    
    # Time domain (0-2s)
    ax1 = fig.add_subplot(gs[0, 0])
    time_mask = t <= 2.0
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        ax1.plot(t[time_mask], targets[i, time_mask], color=color, 
                linewidth=1.5, label=f'{freq}Hz', alpha=0.7)
    ax1.plot(t[time_mask], S_noisy[time_mask], color='black', 
            linewidth=1, alpha=0.5, label='Noisy Mix')
    format_axis(ax1, 'Time (s)', 'Amplitude', 'Time Domain (0-2s)')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Frequency domain (FFT)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_fft_spectrum(ax2, S_noisy, sampling_rate, color='black')
    add_frequency_markers(ax2, frequencies, linewidth=2, alpha=0.7)
    format_axis(ax2, 'Frequency (Hz)', 'Magnitude', 'Frequency Domain (FFT)')
    
    # Spectrogram
    ax3 = fig.add_subplot(gs[1, :])
    f, t_spec, Sxx = scipy_signal.spectrogram(
        S_noisy, fs=sampling_rate, nperseg=512, noverlap=256
    )
    im = ax3.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), 
                        shading='gouraud', cmap='viridis')
    for freq in frequencies:
        ax3.axhline(freq, color='red', linestyle='--', alpha=0.7, linewidth=2)
    format_axis(ax3, 'Time (s)', 'Frequency (Hz)', 'Spectrogram')
    ax3.set_ylim(0, 15)
    plt.colorbar(im, ax=ax3, label='Power (dB)')
    
    # Stacked frequencies
    ax4 = fig.add_subplot(gs[2, :])
    time_mask = t <= 1.0
    for i, (freq, color) in enumerate(zip(frequencies, colors)):
        ax4.plot(t[time_mask], targets[i, time_mask] + i * 2.5, 
                color=color, linewidth=2, label=f'{freq}Hz')
    format_axis(ax4, 'Time (s)', 'Amplitude (offset)', 'Individual Pure Frequencies (Stacked)')
    ax4.legend(loc='upper right', fontsize=9)
    
    save_and_close(save_path)

