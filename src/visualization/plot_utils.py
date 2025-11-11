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

