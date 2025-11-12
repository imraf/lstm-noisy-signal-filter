"""Tests for visualization module."""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import os

from src.visualization.plot_utils import (
    BasePlotter, setup_figure, compute_fft, get_positive_spectrum,
    save_and_close, get_default_colors, format_axis, plot_fft_spectrum,
    add_frequency_markers, add_metric_text
)
from src.visualization.signal_plots import (
    plot_time_domain_signals, plot_overlay_signals, plot_training_samples
)
from src.visualization.training_plots import (
    plot_model_io_structure, plot_training_loss
)
from src.visualization.prediction_plots import (
    plot_predictions_vs_actual, plot_error_distribution, plot_scatter_pred_vs_actual
)
from src.visualization.frequency_plots import (
    plot_frequency_spectrum_comparison, plot_long_sequence_predictions,
    plot_per_frequency_metrics
)
from src.visualization.time_domain_plots import (
    plot_time_domain_signals as plot_time_domain,
    plot_overlay_signals as plot_overlay,
    plot_training_samples as plot_samples
)
from src.visualization.freq_domain_plots import (
    plot_frequency_domain_fft, plot_spectrogram, plot_complete_overview
)


class ConcretePlotter(BasePlotter):
    """Concrete implementation for testing abstract BasePlotter."""
    
    def plot(self, *args, **kwargs):
        """Simple plot implementation."""
        self.create_figure(1, 1, (10, 6))
        return self.fig, self.axes


class TestPlotUtils:
    """Test suite for plot_utils module."""
    
    def test_setup_figure(self):
        """Test figure setup."""
        fig, axes = setup_figure(2, 2, (10, 8), 'Test Title')
        assert fig is not None
        assert axes.shape == (2, 2)
        plt.close(fig)
    
    def test_compute_fft(self):
        """Test FFT computation."""
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        fft_values, fft_freq = compute_fft(signal, 1000)
        assert len(fft_values) == len(signal)
        assert len(fft_freq) == len(signal)
    
    def test_get_positive_spectrum(self):
        """Test positive spectrum extraction."""
        signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        fft_values, fft_freq = compute_fft(signal, 1000)
        magnitude, freq_positive = get_positive_spectrum(fft_values, fft_freq)
        assert len(magnitude) < len(fft_values)
        assert np.all(freq_positive > 0)
    
    def test_get_default_colors(self):
        """Test default color palette."""
        colors = get_default_colors()
        assert len(colors) == 4
        assert all(isinstance(c, str) for c in colors)
    
    def test_format_axis(self):
        """Test axis formatting."""
        fig, ax = plt.subplots(1, 1)
        format_axis(ax, 'X Label', 'Y Label', 'Title', grid=True)
        assert ax.get_xlabel() == 'X Label'
        assert ax.get_ylabel() == 'Y Label'
        assert ax.get_title() == 'Title'
        plt.close(fig)
    
    def test_add_frequency_markers(self):
        """Test frequency marker addition."""
        fig, ax = plt.subplots(1, 1)
        frequencies = [1.0, 3.0, 5.0]
        add_frequency_markers(ax, frequencies)
        plt.close(fig)
    
    def test_add_metric_text(self):
        """Test metric text addition."""
        fig, ax = plt.subplots(1, 1)
        add_metric_text(ax, 'MSE: 0.05', position='top_left')
        plt.close(fig)


class TestBasePlotter:
    """Test suite for BasePlotter class."""
    
    def test_base_plotter_init(self):
        """Test BasePlotter initialization."""
        plotter = ConcretePlotter(dpi=150)
        assert plotter.dpi == 150
        assert plotter.fig is None
        assert plotter.axes is None
    
    def test_create_figure(self):
        """Test figure creation."""
        plotter = ConcretePlotter()
        fig, axes = plotter.create_figure(2, 2, (12, 8), 'Test')
        assert fig is not None
        assert plotter.fig is fig
        plt.close(fig)
    
    def test_format_axis_method(self):
        """Test axis formatting method."""
        plotter = ConcretePlotter()
        fig, ax = plt.subplots(1, 1)
        plotter.format_axis(ax, 'X', 'Y', 'Title')
        assert ax.get_xlabel() == 'X'
        plt.close(fig)
    
    def test_get_colors(self):
        """Test color retrieval."""
        plotter = ConcretePlotter()
        colors = plotter.get_colors()
        assert len(colors) == 4
        colors_limited = plotter.get_colors(2)
        assert len(colors_limited) == 2
    
    def test_save_and_close(self):
        """Test save and close."""
        plotter = ConcretePlotter()
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test.png'
            fig, axes = plotter.create_figure()
            plotter.save_and_close(str(save_path))
            assert save_path.exists()


class TestTimeDomainPlots:
    """Test suite for time domain plotting functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        t = np.linspace(0, 10, 10000)
        frequencies = [1.0, 3.0, 5.0, 7.0]
        targets = np.array([np.sin(2 * np.pi * f * t) for f in frequencies])
        S_noisy = np.sum(targets, axis=0) + np.random.normal(0, 0.1, len(t))
        return t, frequencies, targets, S_noisy
    
    def test_plot_time_domain_signals(self, sample_data):
        """Test time domain signal plotting."""
        t, frequencies, targets, S_noisy = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'time_domain.png'
            plot_time_domain(t, frequencies, targets, S_noisy, str(save_path))
            assert save_path.exists()
    
    def test_plot_overlay_signals(self, sample_data):
        """Test overlay signal plotting."""
        t, frequencies, targets, S_noisy = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'overlay.png'
            plot_overlay(t, frequencies, targets, S_noisy, str(save_path))
            assert save_path.exists()
    
    def test_plot_training_samples(self, sample_data):
        """Test training samples plotting."""
        t, frequencies, targets, S_noisy = sample_data
        S = S_noisy[:100]
        target_vals = targets[0, :100]
        one_hot = np.zeros((100, 4))
        one_hot[:, 0] = 1
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'samples.png'
            plot_samples(S, target_vals, one_hot, frequencies, str(save_path), num_samples=20)
            assert save_path.exists()


class TestFreqDomainPlots:
    """Test suite for frequency domain plotting functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        t = np.linspace(0, 10, 10000)
        frequencies = [1.0, 3.0, 5.0, 7.0]
        targets = np.array([np.sin(2 * np.pi * f * t) for f in frequencies])
        S_noisy = np.sum(targets, axis=0) + np.random.normal(0, 0.1, len(t))
        return t, frequencies, targets, S_noisy
    
    def test_plot_frequency_domain_fft(self, sample_data):
        """Test FFT plotting."""
        t, frequencies, targets, S_noisy = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'fft.png'
            plot_frequency_domain_fft(t, frequencies, targets, S_noisy, str(save_path))
            assert save_path.exists()
    
    def test_plot_spectrogram(self, sample_data):
        """Test spectrogram plotting."""
        t, frequencies, targets, S_noisy = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'spectrogram.png'
            plot_spectrogram(t, S_noisy, frequencies, str(save_path))
            assert save_path.exists()
    
    def test_plot_complete_overview(self, sample_data):
        """Test complete overview plotting."""
        t, frequencies, targets, S_noisy = sample_data
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'overview.png'
            plot_complete_overview(t, frequencies, targets, S_noisy, str(save_path))
            assert save_path.exists()


class TestTrainingPlots:
    """Test suite for training plotting functions."""
    
    def test_plot_model_io_structure(self):
        """Test model I/O structure plotting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'model_io.png'
            plot_model_io_structure(str(save_path))
            assert save_path.exists()
    
    def test_plot_training_loss(self):
        """Test training loss plotting."""
        train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        val_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'training_loss.png'
            plot_training_loss(train_losses, val_losses, str(save_path))
            assert save_path.exists()
    
    def test_plot_training_loss_no_validation(self):
        """Test training loss plotting without validation."""
        train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'training_loss_no_val.png'
            plot_training_loss(train_losses, None, str(save_path))
            assert save_path.exists()


class TestPredictionPlots:
    """Test suite for prediction plotting functions."""
    
    @pytest.fixture
    def prediction_data(self):
        """Create sample prediction data."""
        t = np.linspace(0, 10, 10000)
        frequencies = [1.0, 3.0, 5.0, 7.0]
        targets = np.concatenate([np.sin(2 * np.pi * f * t) for f in frequencies])
        predictions = targets + np.random.normal(0, 0.05, len(targets))
        S_noisy = np.sin(2 * np.pi * 1.0 * t)
        return t, predictions, targets, S_noisy, frequencies
    
    def test_plot_predictions_vs_actual(self, prediction_data):
        """Test predictions vs actual plotting."""
        t, predictions, targets, S_noisy, frequencies = prediction_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'pred_vs_actual.png'
            plot_predictions_vs_actual(
                t, predictions[:len(t)], targets[:len(t)], S_noisy,
                0, frequencies[0], str(save_path)
            )
            assert save_path.exists()
    
    def test_plot_error_distribution(self, prediction_data):
        """Test error distribution plotting."""
        t, predictions, targets, S_noisy, frequencies = prediction_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'error_dist.png'
            plot_error_distribution(predictions, targets, frequencies, str(save_path))
            assert save_path.exists()
    
    def test_plot_scatter_pred_vs_actual(self, prediction_data):
        """Test scatter plot of predictions vs actual."""
        t, predictions, targets, S_noisy, frequencies = prediction_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'scatter.png'
            plot_scatter_pred_vs_actual(predictions, targets, frequencies, str(save_path))
            assert save_path.exists()


class TestFrequencyPlots:
    """Test suite for frequency-specific plotting functions."""
    
    @pytest.fixture
    def frequency_data(self):
        """Create sample frequency data."""
        t = np.linspace(0, 10, 10000)
        frequencies = [1.0, 3.0, 5.0, 7.0]
        targets = np.concatenate([np.sin(2 * np.pi * f * t) for f in frequencies])
        predictions = targets + np.random.normal(0, 0.05, len(targets))
        return t, frequencies, targets, predictions
    
    def test_plot_frequency_spectrum_comparison(self, frequency_data):
        """Test frequency spectrum comparison plotting."""
        t, frequencies, targets, predictions = frequency_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'freq_comparison.png'
            plot_frequency_spectrum_comparison(
                t, predictions, targets, frequencies, str(save_path)
            )
            assert save_path.exists()
    
    def test_plot_long_sequence_predictions(self, frequency_data):
        """Test long sequence predictions plotting."""
        t, frequencies, targets, predictions = frequency_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'long_seq.png'
            plot_long_sequence_predictions(
                t, predictions, targets, frequencies, str(save_path)
            )
            assert save_path.exists()
    
    def test_plot_per_frequency_metrics(self, frequency_data):
        """Test per-frequency metrics plotting."""
        t, frequencies, targets, predictions = frequency_data
        # Create metrics dict with int keys
        test_metrics = {i: {'mse': 0.05 + i*0.01, 'mae': 0.15 + i*0.01} 
                        for i in range(len(frequencies))}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'metrics.png'
            plot_per_frequency_metrics(
                test_metrics, frequencies, str(save_path), split_name='Test'
            )
            assert save_path.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_loss_list(self):
        """Test handling of empty loss list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'empty_loss.png'
            with pytest.raises((ValueError, IndexError)):
                plot_training_loss([], None, str(save_path))
    
    def test_mismatched_shapes(self):
        """Test handling of mismatched array shapes."""
        t = np.linspace(0, 1, 100)
        frequencies = [1.0, 3.0]
        targets = np.random.randn(2, 100)
        S_noisy = np.random.randn(50)  # Wrong size
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'mismatched.png'
            with pytest.raises((ValueError, IndexError)):
                plot_overlay(t, frequencies, targets, S_noisy, str(save_path))
    
    def test_invalid_save_path(self):
        """Test handling of invalid save path."""
        t = np.linspace(0, 1, 100)
        frequencies = [1.0]
        targets = np.random.randn(1, 100)
        S_noisy = np.random.randn(100)
        
        invalid_path = '/nonexistent/directory/file.png'
        with pytest.raises((OSError, IOError, FileNotFoundError)):
            plot_overlay(t, frequencies, targets, S_noisy, invalid_path)

