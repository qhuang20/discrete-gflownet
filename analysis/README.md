# Analysis Package

This folder contains analysis tools for GFlowNet mode discovery and visualization.

## Files

### Core Analysis
- **`analysis_run.py`** - Main analysis script for processing GFlowNet checkpoints
  - Loads checkpoint data
  - Performs mode counting and discovery tracking
  - Generates plots and analysis results
  - Supports diverse mode selection

### Diversity Selection
- **`diversity_selection.py`** - Module for selecting diverse modes
  - Multiple diversity metrics (structure, parameters, rewards, topology, combined)
  - Analysis and comparison functions
  - Reusable across different analysis scripts

### Testing and Documentation
- **`test_diverse_modes.py`** - Test script for diversity selection functionality
- **`DIVERSE_MODES_README.md`** - Detailed documentation for diverse modes feature

## Usage

### Running Analysis
```bash
# From the discrete-gflownet directory
python analysis/analysis_run.py \
  --run_dir your_experiment_name \
  --diversity_metric combined \
  --n_diverse 12
```

### Testing Diversity Selection
```bash
# From the discrete-gflownet directory
python analysis/test_diverse_modes.py
```

### Importing in Other Scripts
```python
from analysis.diversity_selection import select_diverse_modes
from analysis import select_diverse_modes  # Alternative import
```

## Features

- **Mode Discovery Tracking** - Track how modes are discovered during training
- **Diverse Mode Selection** - Select distinct modes using various metrics
- **Network Visualization** - Plot network motifs and somite patterns
- **Performance Analysis** - Analyze reward distributions and mode characteristics
- **Comprehensive Output** - Generate plots, text files, and analysis summaries

## Diversity Metrics

1. **Structure** - Based on weight matrix patterns
2. **Parameters** - Based on parameter value clustering
3. **Rewards** - Based on reward distribution
4. **Topology** - Based on network connectivity patterns
5. **Combined** - Combination of structure and reward diversity

## Output Files

- `top_modes_motifs_and_somites.png` - Top modes visualization
- `diverse_modes_{metric}_motifs_and_somites.png` - Diverse modes visualization
- `mode_and_rewards_discovery.png` - Mode discovery progress
- `analysis_results.txt` - Detailed analysis results
- `modes_with_trajectories.pkl` - Saved mode data 