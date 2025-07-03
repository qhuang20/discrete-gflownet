"""
Configuration package for GFlowNet training.

Available configurations:
- BaseConfig: Base configuration with default parameters
- GridEnvConfig: Configuration for GridEnv (all at once approach)
- GridEnv2Config: Configuration for GridEnv2 (divide and conquer approach)

Users can modify config files directly for custom parameters.
"""

from .base_config import BaseConfig
from .gridenv_config import GridEnvConfig
from .gridenv2_config import GridEnv2Config
from .reward_configs import get_reward_function, REWARD_FUNCTIONS

# Dictionary mapping for easy selection
CONFIGS = {
    'base': BaseConfig,
    'gridenv': GridEnvConfig,
    'gridenv2': GridEnv2Config,
}

def get_config(config_name):
    """Get configuration class by name."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available configs: {list(CONFIGS.keys())}")
    return CONFIGS[config_name] 