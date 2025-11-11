"""Environment variable resolution for configuration."""

import os
from typing import Dict, Any


def resolve_env_vars(config: Dict[str, Any]) -> None:
    """Resolve ${ENV_VAR} placeholders with environment variables.
    
    Modifies config dict in-place.
    
    Args:
        config: Configuration dictionary
    """
    _resolve_dict(config)


def _resolve_dict(d: Dict[str, Any]) -> None:
    """Recursively resolve environment variables in dict."""
    for key, value in d.items():
        if isinstance(value, dict):
            _resolve_dict(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            d[key] = os.getenv(env_var, value)

