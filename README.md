# GFlowNet for Genetic Circuits Design

A clean, modular implementation of GFlowNet for genetic circuits design with improved architecture and configuration management.

## 🎯 **Simple Training Interface**

**One script, clean interface**: `python train.py --env gridenv2 --reward somitogenesis`

No more massive argument lists or confusing multiple training scripts!

## 🏗️ **Project Structure**

```
discrete-gflownet/
├── train.py                    #    Main training script (your entry point!)
├── trainer.py                  #    Core training logic & CheckpointManager
├── disc_gflownet/              #    Main package
│   ├── agents/                 # 🤖 Agent implementations (TB, DB, FlowNet)
│   ├── envs/                   # 🌍 Environment implementations
│   ├── nets/                   # 🧠 Neural network architectures
│   └── utils/                  #    Utilities (logging, plotting, caching)
├── configs/                    #    Configuration management
│   ├── __init__.py             #    Environment registry
│   ├── baseenv_config.py       #    Base configuration class
│   ├── gridenv_config.py       #    GridEnv configuration
│   ├── gridenv2_config.py      #    GridEnv2 configuration
│   └── reward_configs.py       #    Reward function registry
├── reward_func/                # 🎁 Reward function implementations
├── analysis/                   #    Analysis tools and scripts
├── scripts/                    #    Utility scripts
│   └── test_configs.py         #    Configuration testing
├── notebooks/                  #    Organized notebooks
│   ├── experiments/            #    Experimental notebooks
│   ├── analysis/               #    Analysis notebooks
│   └── testing/                #    Testing notebooks
├── runs/                       #    Training outputs
└── docs/                       #    Documentation

```

## 🚀 **Quick Start**

**Main Training Script**: `train.py` - Your single entry point for all training!

### Basic Usage
```bash
# Train with default GridEnv and somitogenesis reward
python train.py

# Use specific environment and reward  
python train.py --env gridenv2 --reward somitogenesis

# Override specific settings
python train.py --env gridenv --steps 2000 --workers 16 --device cuda
```

### Available Options
```bash
# Environments
--env {gridenv, gridenv2, base}

# Reward functions  
--reward {coord, oscillator, somitogenesis}

# Common overrides
--device {cpu, cuda}     # Override device
--steps N                # Override training steps
--workers N              # Override worker count
```







## ⚙️ **Configuration System**

### How It Works
1. **Environment configs** in `configs/` define all parameters
2. **Reward functions** in `reward_func/` provide fitness landscapes  
3. **No massive CLI** - just select environment + reward
4. **Easy customization** - edit config files directly

### Example: Customizing Parameters
```python
# Edit configs/gridenv_config.py
class GridEnvConfig(BaseEnvConfig):
    n_train_steps = 5000    # More training steps
    n_workers = 16          # More parallel workers
    method = 'tb'           # Use Trajectory Balance
    device = 'cuda'         # Use GPU
```








## 🧪 **Development Workflow**

### Adding New Environments
1. Create config in `configs/my_env_config.py`
2. Add to `ENVS` registry in `configs/__init__.py`
3. Implement environment in `disc_gflownet/envs/`

### Adding New Reward Functions
1. Implement in `reward_func/my_reward.py`
2. Add to `REWARD_FUNCTIONS` in `configs/reward_configs.py`

### Running Experiments
```bash
# Quick experiments
python train.py --env gridenv --steps 500

# Production runs  
python train.py --env gridenv2 --steps 10000 --workers 32
```




