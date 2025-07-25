# Environment Configuration System for GFlowNet Training

This directory contains environment configuration files for different GFlowNet training setups, replacing the hardcoded arguments in the original training script.

## Available Environment Configurations

### Base Environment Configuration (`baseenv_config.py`)
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
- `somitogenesis_sparsity`: Sparsity-focused somitogenesis reward function

## Usage

### Using the new training script

```bash
# Use default gridenv environment with somitogenesis reward
python train_with_config.py

# Use specific environment and reward
python train_with_config.py --env gridenv2 --reward oscillator

# Use the new sparsity-focused reward function
python train_with_config.py --env gridenv --reward somitogenesis_sparsity

# To change parameters like n_train_steps, device, etc., edit the environment config files directly
# For example, edit configs/gridenv_config.py to change GridEnv parameters

# See all available environments
python train_with_config.py --help
```

### Using environment configs in Python code

```python
from configs import get_env, ENVS
from configs.reward_configs import get_reward_function

# Get an environment configuration class
env_class = get_env('gridenv')
env_config = env_class()

# Get a reward function
reward_func = get_reward_function('somitogenesis')

# List available environments
print("Available environments:", list(ENVS.keys()))
```

## Adding New Environment Configurations

1. Create a new environment config file in the `configs/` directory
2. Inherit from `BaseEnvConfig` and override necessary parameters
3. Add the import and mapping to `__init__.py`

Example:
```python
# configs/my_custom_env_config.py
from .baseenv_config import BaseEnvConfig

class MyCustomEnvConfig(BaseEnvConfig):
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

## Benefits of the Environment Config System

1. **Clean separation**: No more commented-out code blocks in training script
2. **Easy experimentation**: Switch between environment configurations with a single argument
3. **Reproducibility**: Each environment config is self-contained and version-controlled
4. **Extensibility**: Easy to add new environment configurations without modifying core code
5. **Simple interface**: Only `--env` and `--reward` arguments needed
6. **Direct modification**: Edit environment config files directly for parameter changes
7. **Documentation**: Each environment config file documents its purpose and parameters 