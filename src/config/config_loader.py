"""Configuration loader for LSTM Frequency Filter."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from .env_resolver import resolve_env_vars
from .config_validator import validate_config

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize config loader."""
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        
        if config_path is None:
            config_path = self.config_dir / "default.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)
        resolve_env_vars(self.config)
        
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def merge_config(self, other_config_path: str) -> 'ConfigLoader':
        """Merge another config file into current config."""
        other = self._load_yaml(Path(other_config_path))
        self.config = self._deep_merge(self.config, other)
        resolve_env_vars(self.config)
        return self
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value (supports dot notation)."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def validate(self) -> bool:
        """Validate configuration."""
        return validate_config(self.config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(config_path='{self.config_path}')"


# Global config instance
_global_config: Optional[ConfigLoader] = None


def load_config(config_path: Optional[str] = None, 
                merge_experiment: bool = False) -> ConfigLoader:
    """Load configuration from file."""
    global _global_config
    
    config = ConfigLoader(config_path)
    
    if merge_experiment:
        exp_path = config.config_dir / "experiment.yaml"
        if exp_path.exists():
            config.merge_config(str(exp_path))
            logger.info(f"Merged experimental config: {exp_path}")
    
    _global_config = config
    config.validate()
    
    return config


def get_config() -> ConfigLoader:
    """Get global config instance."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config

