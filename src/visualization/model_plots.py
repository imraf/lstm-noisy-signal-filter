"""Model visualization tools - backward compatibility facade.

This module re-exports functions from refactored submodules.
"""

from .training_plots import plot_model_io_structure, plot_training_loss
from .prediction_plots import (
    plot_predictions_vs_actual,
    plot_error_distribution,
    plot_scatter_pred_vs_actual
)
from .frequency_plots import (
    plot_frequency_spectrum_comparison,
    plot_long_sequence_predictions,
    plot_per_frequency_metrics
)

__all__ = [
    'plot_model_io_structure',
    'plot_training_loss',
    'plot_predictions_vs_actual',
    'plot_error_distribution',
    'plot_scatter_pred_vs_actual',
    'plot_frequency_spectrum_comparison',
    'plot_long_sequence_predictions',
    'plot_per_frequency_metrics'
]
