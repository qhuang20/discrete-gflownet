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
    n = int((-1 + (1 + 4*len(state))**0.5) / 2)
    # print(f"State length: {len(state)}, Calculated n_nodes: {n}")
    return n

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
    s_values = np.ones(n_nodes)  # optional (not in use)
    # s_values = state[n_weights+n_nodes:]
    
    return weights, d_values, s_values, n_nodes, n_weights





def draw_network_motif(state, ax=None, node_size=500, max_edge_weight=200):
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
    # Extract network parameters
    weights, d_values, _, n_nodes, _ = extract_network_parameters(state)
    
    weight_matrix = weights_to_matrix(weights)
    G = nx.DiGraph()
    
    # Create two sets of nodes: weight nodes and d_value nodes
    weight_nodes = range(1, n_nodes + 1)
    d_value_nodes = [f'd{i}' for i in range(1, n_nodes + 1)]
    
    # Add all nodes to the graph
    G.add_nodes_from(weight_nodes)
    G.add_nodes_from(d_value_nodes)
    
    
    # Process edges and their properties
    edge_colors_dict = {}
    edge_widths_dict = {}
    edge_styles_dict = {}
    edge_alphas_dict = {}
    
    # Add edges on weight nodes
    for i in range(n_nodes):
        for j in range(n_nodes):
            if weight_matrix[i, j] != 0:
                # Fix edge direction: if wij, node j points to node i
                edge = (j + 1, i + 1)  # Changed direction to match the requirement
                G.add_edge(*edge, weight=weight_matrix[i, j])
                edge_colors_dict[edge] = 'blue' if weight_matrix[i, j] < 0 else 'red'
                edge_styles_dict[edge] = 'solid'
                edge_alphas_dict[edge] = 1.0
                weight = min(abs(weight_matrix[i, j]), max_edge_weight)
                edge_widths_dict[edge] = 0.5 + (weight * 9.5 / max_edge_weight)
    
    # Add edges between d_value nodes and their corresponding weight nodes
    for i in range(n_nodes):
        d_node = f'd{i+1}'
        w_node = i + 1
        d_value = d_values[i]
        if d_value != 0:  # Only add edge if d_value is non-zero
            G.add_edge(d_node, w_node, weight=d_value)
            edge_colors_dict[(d_node, w_node)] = 'blue' if d_value < 0 else 'red'
            edge_styles_dict[(d_node, w_node)] = 'dashed'
            edge_alphas_dict[(d_node, w_node)] = 0.4
            d_weight = min(abs(d_value), max_edge_weight)
            edge_widths_dict[(d_node, w_node)] = 0.5 + (d_weight * 9.5 / max_edge_weight)
    
    edge_colors = [edge_colors_dict[edge] for edge in G.edges]
    edge_widths = [edge_widths_dict[edge] for edge in G.edges]
    edge_styles = [edge_styles_dict[edge] for edge in G.edges]
    edge_alphas = [edge_alphas_dict[edge] for edge in G.edges]
    
    # Use different layouts based on number of nodes
    pos = {}
    if n_nodes == 2:
        # Use spring layout for 2-node networks
        inner_pos = nx.spring_layout({i: None for i in weight_nodes}, k=1, seed=42)
        outer_pos = nx.spring_layout({f'd{i}': None for i in range(1, n_nodes + 1)}, k=1, seed=42)
    else:
        # Use circular layout for larger networks
        inner_pos = nx.circular_layout({i: None for i in weight_nodes})
        outer_pos = nx.circular_layout({f'd{i}': None for i in range(1, n_nodes + 1)})
    
    # Scale the positions
    for node, (x, y) in inner_pos.items():
        pos[node] = (x * 0.5, y * 0.5)
    
    for node, (x, y) in outer_pos.items():
        pos[node] = (x * 0.8, y * 0.8)
    
    
    
    # Draw nodes with different colors and fixed sizes
    node_colors = plt.cm.rainbow(np.linspace(0, 1, n_nodes))  # Generate colors for weight nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=weight_nodes,
                          node_color=node_colors,
                          node_size=node_size,
                          ax=ax)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=d_value_nodes,
                          node_color='lightgray',
                          node_size=node_size * 0.5,
                          ax=ax)
    
    # Draw edges with styles and transparency
    nx.draw_networkx_edges(G, pos,
                          edge_color=edge_colors,
                          width=edge_widths,
                          style=edge_styles,
                          alpha=edge_alphas,
                          arrows=True,
                          arrowsize=18,
                          connectionstyle="arc3,rad=0.2",
                          ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos,
                           {i: str(i) for i in weight_nodes},
                           font_size=15,
                           font_weight='bold',
                           ax=ax)
    nx.draw_networkx_labels(G, pos,
                           {f'd{i}': f'd{i}' for i in range(1, n_nodes + 1)},
                           font_size=12,
                           ax=ax)
    
    return G


def plot_network_motifs_and_somites(test_states_list, save_path=None):
    """
    Plot network motifs and their corresponding somite patterns in a grid layout.
    
    Args:
        test_states_list (list): List of state configurations (weights + d_values + s_values) to visualize
        save_path (str, optional): Path to save the figure. If None, timestamp will be used
    """
    # Grid layout parameters
    node_size = 550
    max_cols = 6
    n_plots = len(test_states_list)
    n_cols = min(max_cols, n_plots)
    n_rows = 2 * ((n_plots + max_cols - 1) // max_cols)
    fig_width = 8
    fig_height = 6
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
        row = 2 * (idx // max_cols)
        
        # Draw the network motif
        G = draw_network_motif(state, ax=axs[row, col], node_size=node_size)
        
        # Create title showing the state values
        weights, d_values, _, n_nodes, _ = extract_network_parameters(state)
        weight_matrix = weights_to_matrix(weights)
        title = f"Motif {idx+1}\n"
        
        # Add weights in matrix form
        for i in range(n_nodes):
            row_str = "  ".join([f"{weight_matrix[i,j]:<6d}" for j in range(n_nodes)])
            title += f"{row_str}\n"
        
        # Add d_values
        d_str = ", ".join([f"{d:<6d}" for d in d_values])
        title += f"d_values: [{d_str}]"
        
        axs[row, col].set_title(title, fontsize=8)
        
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
        save_path = f"network_motifs_and_somites_grid_{int(time.time())}.png"
    plt.savefig(save_path)
    plt.close()
    
    return save_path






if __name__ == "__main__":
    # Example usage with state vectors (weights + d_values)
    # Each state vector contains:
    # - weights: n_nodes * n_nodes values
    # - d_values: n_nodes values
    test_states_list = [
        # Example 1: 3x3 network
        [80, 66, -86,    # weights for node 1
         -32, 37, -7,    # weights for node 2
         -1, -50, 11,    # weights for node 3
         55, -11, -7],   # d_values for nodes 1,2,3
        
        # # Example 2: 4x4 network
        # [100, -50, 30, -20,    # weights for node 1
        #  -60, 80, -40, 25,     # weights for node 2
        #  35, -45, 90, -55,     # weights for node 3
        #  -25, 65, -35, 70,     # weights for node 4
        #  40, -30, 50, -20],    # d_values for nodes 1,2,3,4
        
        # Example 3: 2x2 network
        # [120, -80,        # weights for node 1
        #  -90, 110,        # weights for node 2
        #  25, -15]         # d_values for nodes 1,2
        
        [100, 1, -42, 96, -27, 62],
        [100, 1, -42, 96, 0, -100]
    ]
    
    # Plot the network motifs
    plot_network_motifs_and_somites(test_states_list)




