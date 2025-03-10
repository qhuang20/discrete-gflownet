import sys
import os
import time
import datetime
import pickle
from argparse import ArgumentParser
from pathlib import Path
import heapq

import numpy as np   
import torch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.spatial import cKDTree

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
from graph.graph import plot_network_motifs_and_somites
from graph.dim_reduction import generate_visualizations


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True, help='Directory containing checkpoint file')
parser.add_argument('--trajectory_idx', type=int, default=0, help='Index of trajectory to animate')
parser.add_argument('--reward_threshold', type=float, default=0.1, help='Reward threshold for mode counting')
parser.add_argument('--top_k', type=int, default=10, help='Number of top states to track')
parser.add_argument('--top_m', type=int, default=12, help='Number of top states to plot') 
args = parser.parse_args()






"""Dimension Reduction Analysis"""

n_top_modes = 6
# run_dir = "___top_20250202_180455_fldb_h256_l3_mr0.001_ts6000_d9_s55_er0.05_etFalse"

def analyze_modes(run_dir, modes_dict, save_dir="visualization_modes"):
    """Analyze and visualize modes."""
    print(f"Total number of modes found: {len(modes_dict)}\n")

    # Sort modes by reward
    sorted_modes = sorted(modes_dict.items(), key=lambda x: x[1]['reward'], reverse=True)
    
    # Print top 10 modes details
    print("Top 10 modes:")
    print("-" * 50)
    for i, (state, info) in enumerate(sorted_modes[:10], 1):
        print(f"\nMode {i}:")
        print(f"State: {list(state)}")
        print(f"Reward: {info['reward']:.3f}")
        print(f"Discovered at training step: {info['step']}")
        print("\nTrajectory:")
        for step, (s, r) in enumerate(zip(info['states'], info['rewards'])):
            print(f"Step {step}: State={s}, Reward={r:.3f}")
        print("-" * 30)

    # Extract states and rewards for visualization
    states = np.array([list(state) for state, _ in sorted_modes])
    rewards = np.array([info['reward'] for _, info in sorted_modes])
    
    print(f"States shape: {states.shape}")
    print(f"First two states:\n{states[:2]}")

    # Create and save visualizations
    save_path = os.path.join("runs", run_dir, save_dir)
    os.makedirs(save_path, exist_ok=True)
    generate_visualizations(states, rewards, save_path)
    print(f"\nModes visualization saved to {save_path}")
    
    return sorted_modes

def analyze_trajectories(sorted_modes, n_top_modes, run_dir, save_dir="visualization_trajectories"):
    """Analyze and visualize trajectories from top modes."""
    all_states = []
    all_rewards = []
    
    # Collect trajectories from top N modes
    for i, (state, info) in enumerate(sorted_modes[:n_top_modes]):
        trajectory_states = info['states']
        trajectory_rewards = info['rewards']
        all_states.extend(trajectory_states)
        all_rewards.extend(trajectory_rewards)
        print(f"Mode {i+1} reward: {info['reward']:.3f}")
        print(f"Trajectory length: {len(trajectory_states)}\n")

    # Convert to numpy arrays
    all_states = np.array(all_states)
    all_rewards = np.array(all_rewards)

    # Create and save visualizations
    save_path = os.path.join("runs", run_dir, save_dir)
    os.makedirs(save_path, exist_ok=True)
    embeddings = generate_visualizations(all_states, all_rewards, save_path, 
                                      cmap='copper_r', show_annotations=True)
    print(f"\nCombined trajectory visualization saved to {save_path}")
    
    return embeddings, all_rewards  # Return rewards along with embeddings

# Load modes data
modes_save_path = os.path.join("runs", args.run_dir, "modes_with_trajectories.pkl")
with open(modes_save_path, 'rb') as f:
    modes_dict = pickle.load(f)

# Run analyses
sorted_modes = analyze_modes(args.run_dir, modes_dict)
embeddings, all_rewards = analyze_trajectories(sorted_modes, n_top_modes=n_top_modes, run_dir=args.run_dir)








"""Find indices of top N rewards to analyze"""

n_top_states = 16
query_indices = [94, 40, 152]

def print_pacmap_details(pacmap_data):
    """Print details about PaCMAP embeddings"""
    print("\nPaCMAP Details:")
    print("-" * 50)
    print("PaCMAP Embedding Shape:", pacmap_data['embedding'].shape)
    print("Original Data Shape:", pacmap_data['original'].shape) 
    print("Index Array Shape:", pacmap_data['idx'].shape)
    print("-" * 50)

def print_top_states(pacmap_data, rewards, n_top_states=16):
    """Print information about top N states by reward"""
    top_n_indices = np.argsort(rewards)[-n_top_states:][::-1]
    
    print(f"\nTop {n_top_states} States with Highest Rewards:")
    print("-" * 50)
    for i in top_n_indices:
        idx = pacmap_data['idx'][i]
        original_state = pacmap_data['original'][i]
        reward = rewards[i]
        print(f"Index {idx}:")
        print(f"Original State: {original_state}")
        print(f"Reward: {reward:.3f}")
        print("-" * 30)

def print_specific_indices(pacmap_data, rewards, indices):
    """Print details for specific indices"""
    for index in indices:
        idx_special = pacmap_data['idx'] == index
        i = np.where(idx_special)[0][0]
        print(f"\nDetails for Index {index}:")
        print("-" * 60)
        print(f"Original State: {pacmap_data['original'][i]}")
        print(f"Reward: {rewards[i]:.3f}")
        print(f"PaCMAP Coordinates: {pacmap_data['embedding'][i]}")


# Print analysis results
pacmap_data = embeddings['pacmap']
print_pacmap_details(pacmap_data)
print_top_states(pacmap_data, all_rewards, n_top_states)
print_specific_indices(pacmap_data, all_rewards, query_indices)




