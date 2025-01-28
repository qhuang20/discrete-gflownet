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
from reward_func.evo_devo import coord_reward_func, oscillator_reward_func, somitogenesis_reward_func




def draw_network_motif(state, ax=None, node_size=500, max_edge_weight=200):
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
    num_nodes = int(np.sqrt(len(state)))
    weight_matrix = np.array(state).reshape(num_nodes, num_nodes)    
    G = nx.DiGraph()
    node_colors = plt.cm.rainbow(np.linspace(0, 1, num_nodes)) # Generate evenly spaced colors for nodes
    G.add_nodes_from(range(1, num_nodes + 1))
    
    # Process edges and their properties
    edge_colors_dict = {}
    edge_widths_dict = {}
    for i in range(num_nodes):
        for j in range(num_nodes):
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
            arrows=True, arrowsize=35, edge_color=edge_colors,
            width=edge_widths, connectionstyle="arc3,rad=0.2")
            
    return G


if __name__ == "__main__":
    # Test cases with different weight configurations
    test_weights_list = [
        [126, -125, -56, 107, 105, -126, 100, -11, 175],
        [153, -159, -32, 19, -14, -45, -101, -32, 42],
        [15, -94, -27, -4, 100, -90, -85, -13, 30],
        [150, -162, 145, 19, -20, 10, -104, -29, 65],
        [1, -166, 119, -87, 58, -85, -111, -60, 78],
        [155, -200, 73, -49, 100, -103, -127, -19, 27]
    ]

    # Grid layout parameters
    node_size = 200
    max_cols = 3  # Reduced to fit both motif and somite plots side by side
    n_plots = len(test_weights_list)
    n_cols = min(max_cols, n_plots)
    n_rows = 2 * ((n_plots + max_cols - 1) // max_cols)  # Double rows to fit somite plots

    # Create figure with dynamic grid size
    fig_width = 8  # Width per subplot
    fig_height = 6  # Height per subplot
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width * n_cols, fig_height * n_rows))
    fig.suptitle("Network Motifs and Somite Patterns", fontsize=16)

    # Plot each test case with its corresponding somite pattern
    for idx, weights in enumerate(test_weights_list):
        col = idx % max_cols
        row = 2 * (idx // max_cols)  # Multiply by 2 to leave space for somite plots
        
        # Draw the network motif
        G = draw_network_motif(weights, ax=axs[row, col], node_size=node_size)
        w11, w12, w13, w21, w22, w23, w31, w32, w33 = weights
        title = f"\n\nMotif {idx+1}\nw11={w11}, w12={w12}, w13={w13}\nw21={w21}, w22={w22}, w23={w23}\nw31={w31}, w32={w32}, w33={w33}\n\n"
        axs[row, col].set_title(title, fontsize=8)
        axs[row, col].set_aspect('equal')
        
        # Draw the somite pattern below the motif
        reward = somitogenesis_reward_func(weights, plot=True, ax=axs[row+1, col])
        axs[row+1, col].set_title(f"Somite Pattern (reward: {reward})", fontsize=8)

    # Remove axes for empty subplots
    total_plots = len(test_weights_list)
    for i in range(n_rows):
        for j in range(n_cols):
            if (i//2) * max_cols + j >= total_plots:
                axs[i, j].axis('off')

    plt.tight_layout()
    plt.savefig(f"network_motifs_and_somites_grid_{int(time.time())}.png")
    plt.close()


