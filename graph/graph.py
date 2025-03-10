import sys
import os

# Add the base directory to sys.path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, base_path)


import time
import datetime
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np   
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

from disc_gflownet.utils.setting import set_seed, set_device, tf
from disc_gflownet.utils.plotting import plot_loss_curve
from disc_gflownet.utils.logging import log_arguments, log_training_loop, track_trajectories
from disc_gflownet.utils.cache import LRUCache
from disc_gflownet.agents.tbflownet_agent import TBFlowNetAgent
from disc_gflownet.agents.dbflownet_agent import DBFlowNetAgent
from disc_gflownet.envs.grid_env import GridEnv
from disc_gflownet.envs.set_env import SetEnv

from scipy.integrate import solve_ivp
from reward_func.evo_devo import coord_reward_func, oscillator_reward_func, somitogenesis_reward_func, weights_to_matrix


def calculate_n_nodes_from_state(state):
    # solve quadratic: n^2 + n - len(state) = 0
    return int((-1 + (1 + 4*len(state))**0.5) / 2)


def extract_network_parameters(state):
    """
    Extract weights, d_values, and s_values from the state vector.
    
    Args:
        state: 1D array containing weights and other parameters
        
    Returns:
        tuple: (weights, d_values, s_values, n_nodes, n_weights)
    """
    n_nodes = calculate_n_nodes_from_state(state)
    n_weights = n_nodes * n_nodes
    weights = state[:n_weights]
    d_values = state[n_weights:n_weights+n_nodes]
    # s_values = state[n_weights+n_nodes:]
    s_values = [1.2, 0.8, 0.9]  # s values
    
    return weights, d_values, s_values, n_nodes, n_weights





def draw_network_motif(state, ax=None, node_size=500, max_edge_weight=200):
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
    # Extract network parameters
    weights, d_values, s_values, n_nodes, n_weights = extract_network_parameters(state)
    
    weight_matrix = weights_to_matrix(weights)
    G = nx.DiGraph()
    node_colors = plt.cm.rainbow(np.linspace(0, 1, n_nodes)) # Generate evenly spaced colors for nodes
    G.add_nodes_from(range(1, n_nodes + 1))
    
    # Process edges and their properties
    edge_colors_dict = {}
    edge_widths_dict = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if weight_matrix[i, j] != 0:
                # Fix edge direction: if wij, node i points to node j
                edge = (i + 1, j + 1)  # Changed from (j + 1, i + 1)
                G.add_edge(*edge, weight=weight_matrix[i, j])
                edge_colors_dict[edge] = 'blue' if weight_matrix[i, j] < 0 else 'red'
                # Clip weight to max_edge_weight and scale to width between 0.5 and 5
                weight = min(abs(weight_matrix[i, j]), max_edge_weight)
                edge_widths_dict[edge] = 0.5 + (weight * 9.5 / max_edge_weight)
    
    edge_colors = [edge_colors_dict[edge] for edge in G.edges]
    edge_widths = [edge_widths_dict[edge] for edge in G.edges]
    
    # Draw the graph
    pos = nx.circular_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_size=node_size,
            node_color=node_colors, font_size=15, font_weight='bold',
            arrows=True, arrowsize=15, edge_color=edge_colors,
            width=edge_widths, connectionstyle="arc3,rad=0.2")
            
    return G


def plot_network_motifs_and_somites(test_states_list, save_path=None):
    """
    Plot network motifs and their corresponding somite patterns in a grid layout.
    
    Args:
        test_states_list (list): List of state configurations (weights + d_values + s_values) to visualize
        save_path (str, optional): Path to save the figure. If None, timestamp will be used
    """
    # Grid layout parameters
    node_size = 200
    max_cols = 3  # Reduced to fit both motif and somite plots side by side
    n_plots = len(test_states_list)
    n_cols = min(max_cols, n_plots)
    n_rows = 2 * ((n_plots + max_cols - 1) // max_cols)  # Double rows to fit somite plots
       
    # Create figure with dynamic grid size
    fig_width = 8  # Width per subplot
    fig_height = 6  # Height per subplot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width * n_cols, fig_height * n_rows))
    fig.suptitle("Network Motifs and Somite Patterns", fontsize=16)

    # Handle single row/column cases
    if n_rows == 1:
        axs = np.array([axs])
    if n_cols == 1:
        axs = np.array([axs]).T

    # Plot each test case with its corresponding somite pattern
    for idx, state in enumerate(test_states_list):
        col = idx % max_cols
        row = 2 * (idx // max_cols)  # Multiply by 2 to leave space for somite plots
        
        # Draw the network motif
        G = draw_network_motif(state, ax=axs[row, col], node_size=node_size)
        
        # Create title showing weights in matrix form
        weights, _, _, n_nodes, _ = extract_network_parameters(state)
        weight_matrix = weights_to_matrix(weights)
        title = f"\n\nMotif {idx+1}\n"
        # Print weight matrix with fixed width formatting
        for i in range(n_nodes):
            row_str = "  ".join([f"{weight_matrix[i,j]:<6d}" for j in range(n_nodes)])
            title += f"{row_str}\n"
        
        axs[row, col].set_title(title, fontsize=8)
        axs[row, col].set_aspect('equal')
        
        # Draw the somite pattern below the motif
        reward = somitogenesis_reward_func(state, plot=True, ax=axs[row+1, col])
        axs[row+1, col].set_title(f"Somite Pattern (reward: {reward})", fontsize=8)

    # Remove axes for empty subplots
    total_plots = len(test_states_list)
    for i in range(n_rows):
        for j in range(n_cols):
            if (i//2) * max_cols + j >= total_plots:
                axs[i, j].axis('off')

    plt.tight_layout()
    
    if save_path is None:
        save_path = f"graph/network_motifs_and_somites_grid_{int(time.time())}.png"
    plt.savefig(save_path)
    plt.close()
    
    return save_path






if __name__ == "__main__":
    # Example usage with full state vectors (weights + d_values + s_values)
    test_states_list = [
        # [126, -125, -56, 107, 105, -126, 100, -11, 175, 1, 1, 1, 1, 1, 1],  # 3x3 weights + 3 d_values + 3 s_values
        # [153, -159, -32, 19, -14, -45, -101, -32, 42, 1, 1, 1, 1, 1, 1],
        # [15, -94, -27, -4, 100, -90, -85, -13, 30, 1, 1, 1, 1, 1, 1],
        # [150, -162, 145, 19, -20, 10, -104, -29, 65, 1, 1, 1, 1, 1, 1],
        # [1, -166, 119, -87, 58, -85, -111, -60, 78, 1, 1, 1, 1, 1, 1],
        # [155, -200, 73, -49, 100, -103, -127, -19, 27, 1, 1, 1, 1, 1, 1],
        # [126, -125, -56, 107, 105, -126, 100, -11, 175, 1, 1, 1, 1, 1, 1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    ]
    
    plot_network_motifs_and_somites(test_states_list)




