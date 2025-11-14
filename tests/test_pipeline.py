"""Tests for pipeline modules."""

import pytest
import torch
from pathlib import Path
from src.pipeline.train_pipeline import execute_training_pipeline
from src.pipeline.visualization_pipeline import generate_all_visualizations


class TestTrainPipeline:
    """Test training pipeline."""
    
    def test_pipeline_minimal_epochs(self, tmp_path):
        """Test pipeline with minimal epochs for speed."""
        results = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0, 5.0, 7.0],
            hidden_size=32,  # Smaller for speed
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=2,  # Minimal epochs
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        assert 'train_mse' in results
        assert 'test_mse' in results
        assert 'model' in results
        assert results['train_mse'] >= 0
        assert results['test_mse'] >= 0
        
    def test_pipeline_returns_all_metrics(self, tmp_path):
        """Test that pipeline returns all expected metrics."""
        results = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],  # Fewer frequencies for speed
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        assert 'train_mse' in results
        assert 'test_mse' in results
        assert 'train_freq_metrics' in results
        assert 'test_freq_metrics' in results
        assert 'model' in results
        assert 'history' in results
        
    def test_pipeline_saves_model(self, tmp_path):
        """Test that pipeline saves model checkpoint."""
        results = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        model_dir = tmp_path / "models"
        assert model_dir.exists()
        
        # Check if at least one model file was saved
        model_files = list(model_dir.glob("*.pth"))
        assert len(model_files) > 0
        
    def test_pipeline_with_different_seeds(self, tmp_path):
        """Test pipeline with different random seeds."""
        results1 = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        results2 = execute_training_pipeline(
            train_seed=99,
            test_seed=100,
            frequencies=[1.0, 3.0],
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        # Results should be different due to different data
        assert results1['train_mse'] != results2['train_mse']


class TestVisualizationPipeline:
    """Test visualization pipeline."""
    
    def test_visualization_generation(self, tmp_path):
        """Test that visualization pipeline generates plots."""
        # First run training to get results
        results = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        # Generate visualizations
        generate_all_visualizations(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            results=results,
            save_dir=tmp_path,
            verbose=False
        )
        
        viz_dir = tmp_path / "visualizations"
        assert viz_dir.exists()
        
        # Check if some plots were generated
        plot_files = list(viz_dir.glob("*.png"))
        assert len(plot_files) > 0
        
    def test_visualization_creates_required_plots(self, tmp_path):
        """Test that key visualizations are created."""
        # Run minimal training
        results = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        # Generate visualizations
        generate_all_visualizations(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            results=results,
            save_dir=tmp_path,
            verbose=False
        )
        
        viz_dir = tmp_path / "visualizations"
        
        # Check for key plot files
        expected_plots = [
            "07_training_loss.png",
            "08_predictions_vs_actual.png"
        ]
        
        for plot_name in expected_plots:
            plot_path = viz_dir / plot_name
            assert plot_path.exists(), f"Expected plot {plot_name} not found"
            
    def test_visualization_without_errors(self, tmp_path):
        """Test that visualization pipeline completes without errors."""
        results = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=1,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        try:
            generate_all_visualizations(
                train_seed=11,
                test_seed=42,
                frequencies=[1.0, 3.0],
                results=results,
                save_dir=tmp_path,
                verbose=False
            )
            success = True
        except Exception as e:
            success = False
            pytest.fail(f"Visualization pipeline raised exception: {e}")
            
        assert success


class TestPipelineIntegration:
    """Integration tests for complete pipeline."""
    
    def test_full_pipeline_execution(self, tmp_path):
        """Test complete pipeline from training to visualization."""
        # Execute training
        results = execute_training_pipeline(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            hidden_size=16,
            num_layers=1,
            learning_rate=0.001,
            batch_size=32,
            num_epochs=2,
            device=torch.device('cpu'),
            save_dir=tmp_path,
            verbose=False
        )
        
        # Generate visualizations
        generate_all_visualizations(
            train_seed=11,
            test_seed=42,
            frequencies=[1.0, 3.0],
            results=results,
            save_dir=tmp_path,
            verbose=False
        )
        
        # Verify outputs
        assert (tmp_path / "models").exists()
        assert (tmp_path / "visualizations").exists()
        assert (tmp_path / "datasets").exists()
        
        # Verify model was saved
        assert len(list((tmp_path / "models").glob("*.pth"))) > 0
        
        # Verify plots were created
        assert len(list((tmp_path / "visualizations").glob("*.png"))) > 0

