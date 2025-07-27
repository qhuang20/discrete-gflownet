#!/usr/bin/env python3
"""
Test grid plotting functionality migrated from test_grid.ipynb
This script generates various plots and saves them to timestamped directories in img/
"""

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if 'notebooks' in current_dir:
    # Navigate up to discrete-gflownet project root
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
else:
    # Already in project root or somewhere else
    project_root = current_dir

if project_root not in sys.path:
    sys.path.append(project_root) 
print(f"Project root: {project_root}")

# Create img directory if it doesn't exist
img_dir = os.path.join(current_dir, 'img')
os.makedirs(img_dir, exist_ok=True)

# --- Timestamped output directory for this run ---
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
run_img_dir = os.path.join(img_dir, RUN_TIMESTAMP)
os.makedirs(run_img_dir, exist_ok=True)
print(f"Images will be saved to: {run_img_dir}")

# Import required modules
from reward_func.evo_devo import (
    coord_reward_func, 
    oscillator_reward_func,
    somitogenesis_reward_func, 
    somitogenesis_sol_func, 
    weights_to_matrix, 
    somitogenesis_sparsity_reward_func
)
from graph.graph import draw_network_motif






def matrix_to_weights(matrix):
    """
    Convert a weight matrix back to the linear weights format used by weights_to_matrix.
    This is the reverse operation of weights_to_matrix.
    
    Args:
        matrix: 2D numpy array or list of lists representing the weight matrix
        
    Returns:
        list: Linear weights array that can be used in test_state
    """
    matrix = np.array(matrix)
    n_nodes = matrix.shape[0]
    
    if n_nodes == 1:
        return [int(matrix[0, 0])]
    
    if n_nodes == 2:
        # For 2x2: [w1,w2,w3,w4] -> [[w1,w4],[w3,w2]]
        # So reverse: [[a,b],[c,d]] -> [a,d,c,b]
        return [int(matrix[0,0]), int(matrix[1,1]), int(matrix[1,0]), int(matrix[0,1])]
    
    # For larger matrices, we need to reverse the recursive construction
    # First, extract the (n-1)x(n-1) inner matrix
    inner_matrix = matrix[:n_nodes-1, :n_nodes-1]
    inner_weights = matrix_to_weights(inner_matrix)
    
    # Then add the weights for the nth node
    new_weights = [int(matrix[n_nodes-1, n_nodes-1])]  # Diagonal element first
    
    # Add alternating row and column entries
    for i in range(n_nodes-1):
        new_weights.append(int(matrix[n_nodes-1, i]))  # Last row
        new_weights.append(int(matrix[i, n_nodes-1]))  # Last column
    
    return inner_weights + new_weights

def show_test_state_as_matrix(test_state):
    """
    Show the given test_state converted to matrix form as readable code.
    All weights and d_values are cast to int for consistency.

    Args:
        test_state: list of parameters (weights + d_values + optional s_values)
    """
    # Cast all to int for consistency
    test_state = [int(x) for x in test_state]

    # Calculate number of nodes
    n_nodes = int((-1 + (1 + 4*len(test_state))**0.5) / 2)
    n_weights = n_nodes * n_nodes

    # Extract weights and convert to matrix
    weights = test_state[:n_weights]
    W = weights_to_matrix(weights)

    print("="*60)
    print("CURRENT TEST_STATE AS MATRIX")
    print("="*60)
    print(f"Your test_state has {len(test_state)} parameters for a {n_nodes}x{n_nodes} weight matrix")
    print(f"Weights occupy indices 0-{n_weights-1}, d_values: {n_weights}-{n_weights+n_nodes-1}, s_values: 1 by default")
    print()

    print("# Weight matrix (from Gene -> to Gene):")
    print("# Rows = To Gene, Columns = From Gene")
    print("weight_matrix = np.array([")
    for i in range(n_nodes):
        row_str = "       ["
        for j in range(n_nodes):
            row_str += f"{int(W[i,j]):8d}"
            if j < n_nodes-1:
                row_str += ","
        row_str += "]"
        if i < n_nodes-1:
            row_str += ","
        row_str += f"  # To Gene {i+1}"
        print(row_str)
    print("])")
    print("# Columns:" + "".join([f"{f'Gene{j+1}':>9s}" for j in range(n_nodes)]))
    print("# Other parameters:")
    d_values = [int(x) for x in test_state[n_weights:n_weights+n_nodes]]
    print(f"d_values = {d_values}")
    print()
    
    print("# Verify round-trip conversion:")
    reconstructed_weights = matrix_to_weights(W)
    reconstructed_test_state = reconstructed_weights + d_values

    print(f"Original weights:      {weights}")
    print(f"Reconstructed weights: {reconstructed_weights}")
    print(f"Weights match: {np.allclose(weights, reconstructed_weights)}")
    print(f"Full test_state match: {np.allclose(test_state, reconstructed_test_state)}")

    return W, d_values

def create_test_state_from_matrix(weight_matrix, d_values):
    """
    Create a test_state from a weight matrix and d_values
    
    Args:
        weight_matrix: 2D array representing weights
        d_values: List of d values
        
    Returns:
        list: Complete test_state that can be used in simulations
    """
    weights = matrix_to_weights(weight_matrix)
    return weights + list(d_values)






def test_coord_reward():
    """Test coordinate reward function - same as notebook"""
    print("=== Testing Coordinate Reward Function ===")
    test_state = (50, -53, -57, 8, 9, -6, -117, 81, 8)
    start_time = time.perf_counter_ns()
    test_reward = coord_reward_func(test_state)
    end_time = time.perf_counter_ns()
    print(f"Time taken to run coord_reward_func: {(end_time - start_time)/1e9:.9f} seconds")
    print(f"Test reward for state {test_state}: {test_reward}")
    return test_reward

def get_default_test_state():
    """Return the default test state from the notebook"""
    return [150.68, -120.00, -75.00, 177.39, 154.94, -182.85, 110.91, -176.08, 120.00, -110.00, 20.00, 0.00, -15.00, -55.00, 200.00, 166.46, 0.00, -15.00, -0.00, 160.00, 105.00, 55.00, 95.60, -145.00, 155.00, -94.91, -150.25, 0.00, 54.03, 0.00, 0.00, 0.00, -89.34, -0.00, -0.00, 50.00, 0.00, 35.93, 0.00, -0.00, 0.00, 27.06, 0.00, 34.32, 50.00, -0.00, -0.00, -0.00, -0.00, -200.00, 175.00, 125.00, -130.00, -50.00, 50.00, 0.00]

def generate_comprehensive_plot(params, cell_pos=6, gene_idx=0, plot_all_genes=False, save_suffix=""):
    """
    Generate the comprehensive 4-subplot plot (equivalent to the interactive widget)
    
    Args:
        params: List of parameters (weights + d values)
        cell_pos: Cell position for oscillation plot (0-99)
        gene_idx: Gene index for heatmap (0-based)
        plot_all_genes: Whether to plot all genes in heatmap
        save_suffix: Suffix to add to filename
    """
    # Calculate number of nodes from the length of state vector
    n_nodes = int((-1 + (1 + 4*len(params))**0.5) / 2)
    n_weights = n_nodes * n_nodes
    
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 26))
    
    # Plot 1: Somite pattern and get reward
    start_time = time.perf_counter_ns()
    reward = somitogenesis_sparsity_reward_func(params, plot=True, ax=ax1, 
                                               gene_idx=gene_idx, plot_all_genes=plot_all_genes)
    end_time = time.perf_counter_ns()
    print(f"Reward for somitogenesis: {reward}")
    print(f"Time taken to run somitogenesis_reward_func: {(end_time - start_time)/1e9:.9f} seconds")
    
    # Plot 2: Oscillation diagram for selected cell_pos
    t_sim, cell_trajectory, _ = somitogenesis_sol_func(params, cell_position=cell_pos)
    for i in range(n_nodes):
        ax2.plot(t_sim, cell_trajectory[:, i], label=f'Gene {i+1}', linewidth=2)        
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Gene Concentration')
    ax2.set_title(f'Gene Expression Dynamics - Cell {cell_pos}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Weighted sum for first row
    weights = params[:n_weights]
    W = weights_to_matrix(weights)
    w_first_row = W[0, :]  # First row of weight matrix
    y_t = cell_trajectory @ w_first_row  # Weighted sum
    ax3.plot(t_sim, y_t, label=f'Weighted Sum (Row 0)', linewidth=3, linestyle='--', color='red')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Weighted Sum')
    ax3.set_title(f'Weighted Sum: y(t) = w₁₁x₁(t) + w₁₂x₂(t) + ... - Cell {cell_pos}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Network motif
    draw_network_motif(params, ax=ax4)
    ax4.set_title(f"{n_nodes}-Node Network Motif")
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_plot_cell{cell_pos}_gene{gene_idx}_{timestamp}"
    if save_suffix:
        filename += f"_{save_suffix}"
    filename += ".png"
    
    filepath = os.path.join(run_img_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Comprehensive plot saved to: {filepath}")
    plt.close()
    
    return filepath

def test_single_gene_heatmap(params, gene_idx=0, save_suffix=""):
    """Test single gene heatmap plotting"""
    print(f"\n=== Testing Single Gene Heatmap (Gene {gene_idx+1}) ===")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    reward = somitogenesis_sparsity_reward_func(params, plot=True, ax=ax, gene_idx=gene_idx)
    print(f"Reward: {reward}")
    
    ax.set_title(f"Single Gene Heatmap - Gene {gene_idx+1}")
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"single_gene_heatmap_gene{gene_idx+1}_{timestamp}"
    if save_suffix:
        filename += f"_{save_suffix}"
    filename += ".png"
    
    filepath = os.path.join(run_img_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Single gene heatmap saved to: {filepath}")
    plt.close()
    
    return filepath

def test_all_genes_grid(params, save_suffix=""):
    """Test plotting all genes in a 3x3 grid"""
    print(f"\n=== Testing All Genes Heatmap (3x3 grid) ===")
    
    # Calculate number of nodes
    n_nodes = int((-1 + (1 + 4*len(params))**0.5) / 2)
    
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot each gene in its subplot, leave extra subplots empty
    for i in range(n_rows * n_cols):
        if i < n_nodes:
            somitogenesis_sparsity_reward_func(
                params, plot=True, ax=axes[i], gene_idx=i, plot_all_genes=False
            )
            axes[i].set_title(f"Gene {i+1}")
        else:
            axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"all_genes_grid_{timestamp}"
    if save_suffix:
        filename += f"_{save_suffix}"
    filename += ".png"
    
    filepath = os.path.join(run_img_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"All genes grid plot saved to: {filepath}")
    plt.close()
    
    return filepath

def plot_weight_matrix(params, save_suffix=""):
    """Plot the weight matrix as a heatmap"""
    print(f"\n=== Plotting Weight Matrix ===")
    
    # Calculate number of nodes
    n_nodes = int((-1 + (1 + 4*len(params))**0.5) / 2)
    n_weights = n_nodes * n_nodes
    
    W = weights_to_matrix(params[:n_weights])
    print("Weight Matrix:")
    print(W)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(W, cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, ax=ax, label='Weight Value')
    
    # Add text annotations
    for i in range(n_nodes):
        for j in range(n_nodes):
            text = ax.text(j, i, f'{W[i, j]:.1f}', 
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title(f'{n_nodes}x{n_nodes} Weight Matrix')
    ax.set_xlabel('From Gene')
    ax.set_ylabel('To Gene')
    ax.set_xticks(range(n_nodes))
    ax.set_yticks(range(n_nodes))
    ax.set_xticklabels([f'Gene {i+1}' for i in range(n_nodes)])
    ax.set_yticklabels([f'Gene {i+1}' for i in range(n_nodes)])
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"weight_matrix_{timestamp}"
    if save_suffix:
        filename += f"_{save_suffix}"
    filename += ".png"
    
    filepath = os.path.join(run_img_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Weight matrix plot saved to: {filepath}")
    plt.close()
    
    return filepath

def generate_parameter_sweep_plots(base_params, param_idx, param_range, param_name="param"):
    """
    Generate plots sweeping one parameter across a range of values
    
    Args:
        base_params: Base parameter set
        param_idx: Index of parameter to sweep
        param_range: List/array of values to test
        param_name: Name of parameter for filenames
    """
    print(f"\n=== Generating Parameter Sweep for {param_name} ===")
    
    rewards = []
    filepaths = []
    
    for i, value in enumerate(param_range):
        params = base_params.copy()
        params[param_idx] = value
        
        print(f"Testing {param_name} = {value}")
        
        # Generate comprehensive plot
        filepath = generate_comprehensive_plot(
            params, 
            save_suffix=f"{param_name}_{value:.2f}_sweep_{i:03d}"
        )
        filepaths.append(filepath)
        
        # Calculate reward for tracking
        reward = somitogenesis_sparsity_reward_func(params, plot=False)
        rewards.append(reward)
    
    # Plot reward vs parameter value
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(param_range, rewards, 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel(f'{param_name} Value')
    ax.set_ylabel('Somitogenesis Reward')
    ax.set_title(f'Reward vs {param_name}')
    ax.grid(True, alpha=0.3)
    
    # Save reward plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reward_filename = f"reward_vs_{param_name}_{timestamp}.png"
    reward_filepath = os.path.join(run_img_dir, reward_filename)
    plt.savefig(reward_filepath, dpi=300, bbox_inches='tight')
    print(f"Reward plot saved to: {reward_filepath}")
    plt.close()
    
    return filepaths, rewards, reward_filepath




def main():


    
    # Show current test_state as matrix
    og_test_state = [160.63, -120.00, -75.00, 176.44, 154.94, -182.85, 181.73, -176.08, 120.00, -110.00, 20.00, -79.18, -15.00, -55.00, 200.00, 166.46, 5.00, -15.00, -0.00, 160.00, 105.00, 55.00, 95.60, -145.00, 155.00, -94.91, -150.25, 40, 54.03, 5.00, -11.85, 0.00, -89.34, -0.00, -0.00, 50.00, 0.00, 35.93, 0.00, -0.00, -5.00, 27.06, 0.00, 34.32, 50.00, -0.00, -0.00, -0.00, -0.00, -200.00, 175.00, 125.00, -130.00, -50.00, 50.00, -5.00]
    _, d_values = show_test_state_as_matrix(og_test_state)
    
    # Modify the matrix here as you wish
    # Full 7x7 matrix (no genes removed)
    weight_matrix = np.array([
        [     160,     176,     181,     -79,       0,      40,       0],  # To Gene 1
        [     -75,    -120,     120,     -55,     105,       5,      -5],  # To Gene 2
        [    -182,    -176,     154,     166,      95,       0,       0],  # To Gene 3
        [      20,     -15,     200,    -110,     155,       0,      50],  # To Gene 4
        [     -15,     160,      55,    -145,       5,      50,       0],  # To Gene 5
        [    -150,      54,     -11,     -89,       0,     -94,       0],  # To Gene 6
        [      35,       0,      27,      34,       0,       0,       0]   # To Gene 7
    ])
    # Columns:    Gene1    Gene2    Gene3    Gene4    Gene5    Gene6    Gene7

    # To "remove" a gene, set its row and column to 0.
    
    # To remove Gene 7 (index 6):
    # weight_matrix[6, :] = 0
    # weight_matrix[:, 6] = 0
    
    # To remove Gene 6 (index 5):
    weight_matrix[5, :] = 0
    weight_matrix[:, 5] = 0

    # To remove Gene 5 (index 4):
    # weight_matrix[4, :] = 0
    # weight_matrix[:, 4] = 0

    # To remove Gene 4 (index 3):
    # weight_matrix[3, :] = 0
    # weight_matrix[:, 3] = 0

    # To remove Gene 3 (index 2):
    # weight_matrix[2, :] = 0
    # weight_matrix[:, 2] = 0

    # To remove Gene 2 (index 1):
    # weight_matrix[1, :] = 0
    # weight_matrix[:, 1] = 0

    # To remove Gene 1 (index 0):
    # weight_matrix[0, :] = 0
    # weight_matrix[:, 0] = 0







    # To convert back to test_state format:
    test_state = create_test_state_from_matrix(weight_matrix, d_values)
    
    print("\n" + "="*60)
    print("GENERATING PLOTS WITH ORIGINAL PARAMETERS")
    print("="*60)
    print(f"Using test state with {len(test_state)} parameters")
    print("Test state values:", test_state)
    n_nodes = int((-1 + (1 + 4*len(test_state))**0.5) / 2)
    print(f"Number of nodes: {n_nodes}")
    
    plot_weight_matrix(test_state, save_suffix="default")
    test_single_gene_heatmap(test_state, gene_idx=0, save_suffix="default")
    test_all_genes_grid(test_state, save_suffix="default")
    # Generate comprehensive plot with default parameters
    generate_comprehensive_plot(test_state, cell_pos=6, gene_idx=0, save_suffix="default")
        
    # Example parameter sweep (uncomment to run)
    # print(f"\n=== Example Parameter Sweep ===")
    # param_range = np.linspace(-200, 200, 11)  # 11 values from -200 to 200
    # generate_parameter_sweep_plots(test_state, 0, param_range, "weight_1")
    
    print("\n" + "="*60)
    print("All plots generated successfully!")
    print(f"Check the {run_img_dir} directory for saved plots.")

if __name__ == "__main__":
    main() 