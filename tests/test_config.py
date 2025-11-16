"""Tests for configuration module."""

import pytest
import yaml
import tempfile
from pathlib import Path
from src.config.config_loader import ConfigLoader, load_config
from src.config.config_validator import validate_config
from src.config.env_resolver import resolve_env_vars


class TestConfigLoader:
    """Test configuration loading functionality."""
    
    def test_load_default_config(self):
        """Test loading default configuration."""
        config = ConfigLoader()
        
        assert config is not None
        assert config.config is not None
        assert 'model' in config.config
        assert 'training' in config.config
        assert 'data' in config.config
        
    def test_load_specific_config(self):
        """Test loading specific configuration file."""
        config = ConfigLoader('config/default.yaml')
        
        assert config is not None
        assert 'model' in config.config
        
    def test_get_nested_value(self):
        """Test getting nested configuration values."""
        config = ConfigLoader()
        
        hidden_size = config.get('model.hidden_size', default=32)
        assert hidden_size == 64
        
        learning_rate = config.get('training.learning_rate', default=0.01)
        assert learning_rate == 0.001
        
    def test_get_with_default(self):
        """Test getting value with default fallback."""
        config = ConfigLoader()
        
        missing_value = config.get('nonexistent.key', default=42)
        assert missing_value == 42
        
    def test_merge_configs(self):
        """Test merging configurations."""
        config = ConfigLoader('config/default.yaml')
        config.merge_config('config/experiment.yaml')
        
        assert config.config is not None
        assert 'model' in config.config
        
    def test_load_nonexistent_file(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader('nonexistent.yaml')
    
    def test_set_value(self):
        """Test setting configuration value."""
        config = ConfigLoader()
        config.set('model.hidden_size', 128)
        
        assert config.get('model.hidden_size') == 128
        
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = ConfigLoader()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert 'training' in config_dict


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = {
            'model': {'hidden_size': 64, 'num_layers': 2},
            'training': {'num_epochs': 100, 'batch_size': 32, 'learning_rate': 0.001},
            'data': {'train_seed': 11, 'test_seed': 42, 'frequencies': [1, 3, 5, 7]}
        }
        
        is_valid = validate_config(config)
        
        assert is_valid
        
    def test_validate_missing_required_field(self):
        """Test validation with missing required fields."""
        config = {
            'model': {'hidden_size': 64},
            'training': {'num_epochs': 100}
        }
        
        with pytest.raises(ValueError):
            validate_config(config)
        
    def test_validate_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        config = {
            'model': {'hidden_size': 64, 'num_layers': 2},
            'training': {'num_epochs': 100, 'batch_size': 32, 'learning_rate': -0.001},
            'data': {'train_seed': 11, 'test_seed': 42, 'frequencies': [1, 3, 5, 7]}
        }
        
        with pytest.raises(ValueError):
            validate_config(config)
        
    def test_validate_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        config = {
            'model': {'hidden_size': 64, 'num_layers': 2},
            'training': {'num_epochs': 100, 'batch_size': -32, 'learning_rate': 0.001},
            'data': {'train_seed': 11, 'test_seed': 42, 'frequencies': [1, 3, 5, 7]}
        }
        
        with pytest.raises(ValueError):
            validate_config(config)
        
    def test_validate_invalid_hidden_size(self):
        """Test validation with invalid hidden size."""
        config = {
            'model': {'hidden_size': -10, 'num_layers': 2},
            'training': {'num_epochs': 100, 'batch_size': 32, 'learning_rate': 0.001},
            'data': {'train_seed': 11, 'test_seed': 42, 'frequencies': [1, 3, 5, 7]}
        }
        
        with pytest.raises(ValueError):
            validate_config(config)


class TestEnvResolver:
    """Test environment variable resolution."""
    
    def test_resolve_env_variables(self, monkeypatch):
        """Test resolving environment variables in config."""
        monkeypatch.setenv('HIDDEN_SIZE', '128')
        monkeypatch.setenv('LEARNING_RATE', '0.002')
        
        config = {
            'model': {'hidden_size': '${HIDDEN_SIZE}'},
            'training': {'learning_rate': '${LEARNING_RATE}'}
        }
        
        resolve_env_vars(config)
        
        assert config['model']['hidden_size'] == '128'
        assert config['training']['learning_rate'] == '0.002'
        
    def test_resolve_nested_env_variables(self, monkeypatch):
        """Test resolving nested environment variables."""
        monkeypatch.setenv('OUTPUT_DIR', '/tmp/outputs')
        
        config = {
            'output': {
                'base_dir': '${OUTPUT_DIR}',
                'model_dir': '${OUTPUT_DIR}'
            }
        }
        
        resolve_env_vars(config)
        
        assert config['output']['base_dir'] == '/tmp/outputs'
        
    def test_no_env_variables(self):
        """Test config without environment variables."""
        config = {
            'model': {'hidden_size': 64},
            'training': {'learning_rate': 0.001}
        }
        
        config_copy = config.copy()
        resolve_env_vars(config)
        
        # Should remain unchanged
        assert config['model']['hidden_size'] == config_copy['model']['hidden_size']


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_load_and_validate_default_config(self):
        """Test loading and validating default config."""
        config = ConfigLoader()
        is_valid = config.validate()
        
        assert is_valid
        
    def test_load_merge_and_validate(self):
        """Test loading, merging, and validating configs."""
        config = load_config('config/default.yaml', merge_experiment=True)
        
        # Merged config should still be valid
        assert config is not None
        assert config.config is not None

