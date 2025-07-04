"""
Environment configuration package for GFlowNet training.

Available configurations:
- BaseEnvConfig: Base configuration with default parameters
- GridEnvConfig: Configuration for GridEnv (all at once approach)
- GridEnv2Config: Configuration for GridEnv2 (divide and conquer approach)

Users can modify config files directly for custom parameters.
"""

from .baseenv_config import BaseEnvConfig
from .gridenv_config import GridEnvConfig
from .gridenv2_config import GridEnv2Config
from .reward_configs import get_reward_function, REWARD_FUNCTIONS

# Dictionary mapping for easy selection
ENVS = {
    'base': BaseEnvConfig,
    'gridenv': GridEnvConfig,
    'gridenv2': GridEnv2Config,
}

def get_env(env_name):
    """Get environment configuration class by name."""
    if env_name not in ENVS:
        raise ValueError(f"Unknown environment: {env_name}. Available environments: {list(ENVS.keys())}")
    return ENVS[env_name] 