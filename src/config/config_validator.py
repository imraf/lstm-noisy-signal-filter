"""Configuration validation."""

from typing import Dict, Any, List


REQUIRED_KEYS = [
    'data.train_seed',
    'data.test_seed',
    'data.frequencies',
    'model.hidden_size',
    'model.num_layers',
    'training.num_epochs',
    'training.batch_size',
    'training.learning_rate'
]


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If configuration is invalid
    """
    for key in REQUIRED_KEYS:
        if _get_nested_value(config, key) is None:
            raise ValueError(f"Required configuration key missing: {key}")
    
    learning_rate = _get_nested_value(config, 'training.learning_rate')
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    
    batch_size = _get_nested_value(config, 'training.batch_size')
    if batch_size is not None and batch_size <= 0:
        raise ValueError("Batch size must be positive")
    
    hidden_size = _get_nested_value(config, 'model.hidden_size')
    if hidden_size is not None and hidden_size <= 0:
        raise ValueError("Hidden size must be positive")
    
    return True


def _get_nested_value(d: Dict[str, Any], key: str) -> Any:
    """Get nested value from dict using dot notation."""
    keys = key.split('.')
    value = d
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return None
    return value

