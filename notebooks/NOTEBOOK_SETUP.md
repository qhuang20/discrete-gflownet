# Notebook Setup Guide

When creating new notebooks in this project, always add this import setup code to your first cell to ensure all modules can be imported correctly:

## 📋 **Copy-Paste Setup Code**

```python
import sys
import os

# Add the project root directory to sys.path
current_dir = os.getcwd()
if 'notebooks' in current_dir:
    # Navigate up to project root
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
else:
    # Already in project root or somewhere else
    project_root = current_dir

if project_root not in sys.path:
    sys.path.append(project_root)

print(f"Project root: {project_root}")

# Now you can import from the project
from configs import get_env, get_reward_function
from reward_func.evo_devo import somitogenesis_reward_func
from disc_gflownet.envs.grid_env import GridEnv
# ... other imports as needed
```

## 🎯 **Why This is Needed**

When notebooks are in subdirectories like `notebooks/testing/` or `notebooks/analysis/`, Python can't find the project modules unless we explicitly add the project root to the Python path.

## 📁 **Directory Structure**

```
discrete-gflownet/           # ← Project root  
├── configs/                 # ← Modules we want to import
├── disc_gflownet/          # ← Modules we want to import  
├── reward_func/            # ← Modules we want to import
└── notebooks/              
    ├── testing/            # ← Your notebook might be here
    ├── analysis/           # ← Or here
    └── experiments/        # ← Or here
```

## ✅ **Verification**

After running the setup code, you should see output like:
```
Project root: /path/to/discrete-gflownet
```

If you see a path ending in `/notebooks/testing` or similar, the setup didn't work correctly.

## 🚀 **Quick Test**

To verify everything works, try importing a key module:

```python
# This should work without errors
from configs import ENVS
print(f"Available environments: {list(ENVS.keys())}")
```

You should see:
```
Available environments: ['base', 'gridenv', 'gridenv2']
``` 