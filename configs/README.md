# Configuration System for GFlowNet Training

This directory contains configuration files for different GFlowNet training setups, replacing the hardcoded arguments in the original training script.

## Available Configurations

### Base Configuration (`base_config.py`)
- Contains default parameters shared across all environments
- Training settings (device, seed, training steps, etc.)
- Model parameters (method, learning rates, architecture, etc.)
- Environment base settings

### Environment Configurations

#### `gridenv_config.py` 
- Configuration for GridEnv (all at once approach)
- Parameters: `n_dims=56`, `max_nodes=2`, `max_edges=4`, `n_steps=23`
- Modify this file directly for custom GridEnv parameters

#### `gridenv2_config.py`
- Configuration for GridEnv2 (divide and conquer approach)
- Parameters: `n_dims=12`, `n_steps=72`, `steps_per_network={1:8, 2:24, 3:40}`
- Modify this file directly for custom GridEnv2 parameters

## Reward Functions (`reward_configs.py`)

Available reward functions:
- `coord`: Coordinate reward function
- `oscillator`: Oscillator reward function  
- `somitogenesis`: Somitogenesis reward function (default)

## Usage

### Using the new training script

```bash
# Use default gridenv config with somitogenesis reward
python train_with_config.py

# Use specific config and reward
python train_with_config.py --config gridenv2 --reward oscillator

# To change parameters like n_train_steps, device, etc., edit the config files directly
# For example, edit configs/gridenv_config.py to change GridEnv parameters

# See all available configs
python train_with_config.py --help
```

### Using configs in Python code

```python
from configs import get_config, CONFIGS
from configs.reward_configs import get_reward_function

# Get a configuration class
config_class = get_config('gridenv')
config = config_class()

# Get a reward function
reward_func = get_reward_function('somitogenesis')

# List available configs
print("Available configs:", list(CONFIGS.keys()))
```

## Adding New Configurations

1. Create a new config file in the `configs/` directory
2. Inherit from `BaseConfig` and override necessary parameters
3. Add the import and mapping to `__init__.py`

Example:
```python
# configs/my_custom_config.py
from .base_config import BaseConfig

class MyCustomConfig(BaseConfig):
    # Training parameters
    n_train_steps = 5000
    method = 'tb'
    device = 'cuda'
    
    # Environment parameters  
    n_dims = 100
    n_steps = 50
    env_type = 'GridEnv'
    
    # ... other parameters as needed
```

## Benefits of the Config System

1. **Clean separation**: No more commented-out code blocks in training script
2. **Easy experimentation**: Switch between configurations with a single argument
3. **Reproducibility**: Each config is self-contained and version-controlled
4. **Extensibility**: Easy to add new configurations without modifying core code
5. **Simple interface**: Only `--config` and `--reward` arguments needed
6. **Direct modification**: Edit config files directly for parameter changes
7. **Documentation**: Each config file documents its purpose and parameters 