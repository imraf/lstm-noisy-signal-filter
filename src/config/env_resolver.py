"""Environment variable resolution for configuration."""

import os
from typing import Dict, Any
from pathlib import Path


def load_env_file(env_path: str = None) -> None:
    """Load environment variables from .env file.
    
    Args:
        env_path: Path to .env file (defaults to .env in project root)
    """
    try:
        from dotenv import load_dotenv
        if env_path is None:
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / ".env"
        
        if Path(env_path).exists():
            load_dotenv(env_path)
    except ImportError:
        pass


def resolve_env_vars(config: Dict[str, Any]) -> None:
    """Resolve ${ENV_VAR} placeholders and apply environment variable overrides.
    
    Modifies config dict in-place.
    
    Priority order:
    1. Environment variables directly matching config keys
    2. ${ENV_VAR} placeholders in config values
    
    Args:
        config: Configuration dictionary
    """
    load_env_file()
    _resolve_dict(config)
    _apply_env_overrides(config)


def _resolve_dict(d: Dict[str, Any]) -> None:
    """Recursively resolve environment variables in dict."""
    for key, value in d.items():
        if isinstance(value, dict):
            _resolve_dict(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            d[key] = os.getenv(env_var, value)


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """Apply direct environment variable overrides to config.
    
    Looks for environment variables matching config structure.
    Examples:
        DEVICE -> config['device']
        LEARNING_RATE -> config['training']['learning_rate']
    """
    env_mappings = {
        'DEVICE': 'device',
        'LOG_LEVEL': 'logging.level',
        'OUTPUT_DIR': 'output.dir',
        'MODEL_DIR': 'output.model_dir',
        'DATASET_DIR': 'output.dataset_dir',
        'VISUALIZATION_DIR': 'output.visualization_dir',
        'RANDOM_SEED': 'data.train_seed',
        'PLOT_DPI': 'visualization.dpi',
        'PLOT_FORMAT': 'visualization.format',
        'NUM_WORKERS': 'training.num_workers',
        'EXPERIMENT_NAME': 'experiment.name',
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            _set_nested_value(config, config_path, _parse_value(value))


def _set_nested_value(config: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested config value using dot notation."""
    keys = path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate type."""
    value_lower = value.lower()
    
    if value_lower in ('true', 'yes', '1'):
        return True
    elif value_lower in ('false', 'no', '0'):
        return False
    elif value.isdigit():
        return int(value)
    
    try:
        return float(value)
    except ValueError:
        return value

