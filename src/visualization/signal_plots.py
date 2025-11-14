"""Signal visualization tools - backward compatibility facade.

This module re-exports functions from refactored submodules.
"""

from .time_domain_plots import (
    plot_time_domain_signals,
    plot_overlay_signals,
    plot_training_samples
)
from .freq_domain_plots import (
    plot_frequency_domain_fft,
    plot_spectrogram,
    plot_complete_overview
)

__all__ = [
    'plot_time_domain_signals',
    'plot_overlay_signals',
    'plot_training_samples',
    'plot_frequency_domain_fft',
    'plot_spectrogram',
    'plot_complete_overview'
]
