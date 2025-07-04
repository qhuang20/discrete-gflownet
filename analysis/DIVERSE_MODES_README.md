# Diverse Modes Selection

This document explains the new diverse modes selection functionality added to the analysis script.

## Overview

When analyzing discovered modes in GFlowNet training, you often see many similar network structures in the top-performing modes. The diverse modes selection feature helps identify and visualize modes that are distinct from each other, providing a better understanding of the variety of solutions discovered by the model.

## Usage

### Command Line Arguments

The analysis script now supports additional arguments for diverse modes selection:

```bash
python analysis_run.py --run_dir <run_directory> \
    --diversity_metric <metric> \
    --n_diverse <number> \
    --top_m <number>
```

**New Arguments:**
- `--diversity_metric`: Choose the diversity metric to use
  - `structure`: Based on weight matrix patterns
  - `parameters`: Based on parameter value clustering
  - `rewards`: Based on reward distribution
  - `topology`: Based on network connectivity patterns
  - `combined`: Combination of structure and reward diversity (default)
- `--n_diverse`: Number of diverse modes to select and plot (default: 24)

### Example Usage

```bash
# Select 12 diverse modes using topology-based diversity
python analysis_run.py --run_dir my_experiment \
    --diversity_metric topology \
    --n_diverse 12

# Select 20 diverse modes using combined diversity metric
python analysis_run.py --run_dir my_experiment \
    --diversity_metric combined \
    --n_diverse 20
```

## Diversity Metrics

### 1. Structure Diversity (`structure`)
- **What it does**: Compares weight matrices using cosine similarity
- **Best for**: Finding modes with different interaction strengths and patterns
- **Use case**: When you want to see how different the actual network connections are

### 2. Parameter Diversity (`parameters`)
- **What it does**: Uses K-means clustering on the full parameter vectors
- **Best for**: Finding modes that are distant in the full parameter space
- **Use case**: When you want to see the full range of parameter combinations

### 3. Reward Diversity (`rewards`)
- **What it does**: Bins modes by reward values and selects from different bins
- **Best for**: Ensuring representation across the reward spectrum
- **Use case**: When you want to see both high and low performing diverse solutions

### 4. Topology Diversity (`topology`)
- **What it does**: Compares binary connectivity patterns (which connections exist)
- **Best for**: Finding modes with different network architectures
- **Use case**: When you want to see different types of network structures

### 5. Combined Diversity (`combined`)
- **What it does**: Combines structure and reward diversity approaches
- **Best for**: Balanced selection considering both performance and structure
- **Use case**: General purpose diverse mode selection

## Output Files

The script generates additional output files:

1. **Diverse Modes Plot**: `diverse_modes_{metric}_motifs_and_somites.png`
   - Shows network motifs and somite patterns for diverse modes
   - Filename includes the diversity metric used

2. **Analysis Results**: Updated `analysis_results.txt`
   - Contains information about selected diverse modes
   - Shows rewards and discovery steps for each diverse mode

## Testing

You can test the diverse modes functionality using the provided test script:

```bash
python test_diverse_modes.py
```

This script:
- Creates synthetic test modes with different characteristics
- Tests all diversity metrics
- Generates comparison plots
- Shows how different metrics perform

## Interpreting Results

### When to Use Each Metric

- **High similarity in top modes**: Use `topology` or `structure`
- **Want to see full parameter range**: Use `parameters`
- **Want to see reward distribution**: Use `rewards`
- **General exploration**: Use `combined`

### What to Look For

1. **Network Structure Differences**:
   - Different connectivity patterns
   - Varying interaction strengths
   - Different types of feedback loops

2. **Performance Patterns**:
   - How diverse modes perform compared to top modes
   - Whether high diversity correlates with performance
   - Trade-offs between diversity and reward

3. **Discovery Insights**:
   - When diverse modes were discovered during training
   - Whether the model explores different solution types
   - Convergence patterns in diverse solutions

## Example Analysis Workflow

1. **Start with combined diversity**:
   ```bash
   python analysis_run.py --run_dir experiment --diversity_metric combined --n_diverse 12
   ```

2. **Examine the results** and identify what aspects you want to explore further

3. **Try specific metrics** based on your observations:
   - If networks look similar: try `topology`
   - If you want parameter range: try `parameters`
   - If you want reward distribution: try `rewards`

4. **Compare with top modes** to understand the relationship between performance and diversity

## Tips

- Start with a smaller number of diverse modes (6-12) for easier interpretation
- Compare diverse modes with top modes to understand the trade-offs
- Use the test script to understand how different metrics work
- Consider the biological interpretation of different network structures
- Look for patterns in when diverse modes were discovered during training 