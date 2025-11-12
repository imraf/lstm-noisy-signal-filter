"""Common plotting utilities for visualizations."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from typing import Tuple


def setup_figure(nrows: int, ncols: int, figsize: Tuple[int, int], title: str = None):
    """Create figure with axes."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    return fig, axes


def compute_fft(signal: np.ndarray, sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT and frequencies.
    
    Args:
        signal: Input signal
        sampling_rate: Sampling rate
    
    Returns:
        fft_values: FFT values
        fft_freq: FFT frequencies
    """
    fft_values = fft(signal)
    fft_freq = fftfreq(len(signal), 1/sampling_rate)
    return fft_values, fft_freq


def get_positive_spectrum(fft_values: np.ndarray, fft_freq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get positive frequency spectrum.
    
    Args:
        fft_values: FFT values
        fft_freq: FFT frequencies
    
    Returns:
        magnitude: Magnitude spectrum (positive frequencies)
        freq_positive: Positive frequencies
    """
    positive_mask = fft_freq > 0
    magnitude = 2.0/len(fft_values) * np.abs(fft_values[positive_mask])
    freq_positive = fft_freq[positive_mask]
    return magnitude, freq_positive


def save_and_close(save_path: str, dpi: int = 300):
    """Save figure and close."""
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def get_default_colors():
    """Get default color palette."""
    return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def format_axis(ax, xlabel: str = None, ylabel: str = None, title: str = None, grid: bool = True):
    """Format axis with labels and title."""
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold')
    if grid:
        ax.grid(True, alpha=0.3)


def plot_fft_spectrum(ax, signal: np.ndarray, sampling_rate: float, color: str = 'black', 
                      linewidth: float = 1.5, xlim: Tuple[float, float] = (0, 15)):
    """Plot FFT spectrum on given axis.
    
    Args:
        ax: Matplotlib axis
        signal: Input signal
        sampling_rate: Sampling rate
        color: Line color
        linewidth: Line width
        xlim: X-axis limits
    """
    fft_vals, fft_freq = compute_fft(signal, sampling_rate)
    magnitude, freq_positive = get_positive_spectrum(fft_vals, fft_freq)
    ax.plot(freq_positive, np.abs(fft_vals[fft_freq > 0]), color=color, linewidth=linewidth)
    ax.set_xlim(xlim)
    ax.grid(True, alpha=0.3)


def add_frequency_markers(ax, frequencies, color: str = 'red', linestyle: str = '--', 
                          alpha: float = 0.5, linewidth: float = 1):
    """Add vertical lines for target frequencies.
    
    Args:
        ax: Matplotlib axis
        frequencies: List of frequencies to mark
        color: Line color
        linestyle: Line style
        alpha: Line transparency
        linewidth: Line width
    """
    for freq in frequencies:
        ax.axvline(freq, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth)


def add_metric_text(ax, text: str, position: str = 'top_left'):
    """Add metric text box to axis.
    
    Args:
        ax: Matplotlib axis
        text: Text to display
        position: Position ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    """
    positions = {
        'top_left': (0.02, 0.98, 'top'),
        'top_right': (0.98, 0.98, 'top'),
        'bottom_left': (0.02, 0.02, 'bottom'),
        'bottom_right': (0.98, 0.02, 'bottom')
    }
    x, y, va = positions.get(position, positions['top_left'])
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, verticalalignment=va,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

