# GFlowNet for Genetic Circuits Design

A clean, modular implementation of GFlowNet for genetic circuits design with improved architecture and configuration management.

## 🎯 **Simple Training Interface**

**One script, clean interface**: `python train.py --env gridenv2 --reward somitogenesis`

No more massive argument lists or confusing multiple training scripts!

## 🏗️ **Project Structure**

```
discrete-gflownet/
├── train.py                    # 🚀 Main training script (your entry point!)
├── trainer.py                  # 🎯 Core training logic & CheckpointManager
├── disc_gflownet/              # 📦 Main package
│   ├── agents/                 # 🤖 Agent implementations (TB, DB, FlowNet)
│   ├── envs/                   # 🌍 Environment implementations
│   ├── nets/                   # 🧠 Neural network architectures
│   └── utils/                  # 🛠️ Utilities (logging, plotting, caching)
├── configs/                    # ⚙️ Configuration management
│   ├── __init__.py             # 📋 Environment registry
│   ├── baseenv_config.py       # 🔧 Base configuration class
│   ├── gridenv_config.py       # 🏠 GridEnv configuration
│   ├── gridenv2_config.py      # 🏠 GridEnv2 configuration
│   └── reward_configs.py       # 🎁 Reward function registry
├── reward_func/                # 🎯 Reward function implementations
├── analysis/                   # 📊 Analysis tools and scripts
├── scripts/                    # 🔧 Utility scripts
│   └── test_configs.py         # 🧪 Configuration testing
├── notebooks/                  # 📓 Organized notebooks
│   ├── experiments/            # 🧪 Experimental notebooks
│   ├── analysis/              # 📊 Analysis notebooks
│   └── testing/               # 🔍 Testing notebooks
├── runs/                      # 💾 Training outputs
└── docs/                      # 📚 Documentation

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

## 🏗️ **Architecture Improvements**

### Before vs After

| **Before (Old train.py)** | **After (Refactored)** |
|---------------------------|------------------------|
| ❌ 248 lines with massive arg parser | ✅ Clean 80-line main script |
| ❌ Global variables (`global losses, zs`) | ✅ Encapsulated in `GFlowNetTrainer` class |  
| ❌ Monolithic `main()` function | ✅ Modular trainer with clear responsibilities |
| ❌ Hardcoded env selection | ✅ Dynamic environment registry |
| ❌ Commented-out config blocks | ✅ Clean config class hierarchy |
| ❌ Mixed training/logging logic | ✅ Separate `CheckpointManager` class |

### Key Classes

#### `GFlowNetTrainer`
- **Encapsulates** all training state (no globals!)
- **Modular setup** methods for env/agent/logging
- **Clean training loop** with step-by-step logic
- **Robust error handling** with checkpoint recovery

#### `CheckpointManager`  
- **Dedicated checkpoint** saving/loading
- **Interruption handling** for graceful shutdowns
- **Consistent state** preservation across runs

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

## 📊 **Notebooks Organization**

- **`notebooks/experiments/`** - New experiment notebooks
- **`notebooks/analysis/`** - Data analysis and visualization  
- **`notebooks/testing/`** - Testing and validation notebooks

**Note**: All existing notebooks have been updated with proper import paths. For new notebooks, see `notebooks/NOTEBOOK_SETUP.md` for the required setup code.

## 🔧 **CLI Memory**

The system remembers your preferences. Based on your usage patterns, it will use `--env` instead of `--config` for environment specification.

## 🎯 **Benefits of This Refactor**

1. **🧹 Clean separation** - No more commented-out code blocks
2. **🔄 Easy experimentation** - Switch environments with single flag
3. **📋 Reproducibility** - Each config is self-contained
4. **🔧 Extensibility** - Add new environments without touching core code
5. **⚡ Performance** - Better organized multiprocessing
6. **🛡️ Robustness** - Proper error handling and state management
7. **📚 Maintainability** - Clear class structure and responsibilities

## 🔮 **Usage Examples**

```bash
# Quick GridEnv experiment
python train.py --env gridenv --steps 1000

# Your preferred interface (works perfectly!)
python train.py --env gridenv2 --reward somitogenesis

# Long GridEnv2 training with multiprocessing  
python train.py --env gridenv2 --steps 10000 --workers 20

# GPU accelerated training
python train.py --env gridenv --device cuda --steps 5000

# Different reward landscapes
python train.py --reward coord --steps 2000
python train.py --reward oscillator --steps 2000  
```

Your research just got a lot more organized! 🚀 