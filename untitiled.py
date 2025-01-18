import sys
import os
import time
import datetime
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np   
import torch
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

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

# Configuration
TOP_N = 20  # Number of top states to analyze

# Load checkpoint
checkpoint_path = "runs/20250118_145704_fldb_h256_l3_mr0.001_ts2000_d2_s12_er0.35_etFalse/checkpoint_interrupted.pt"
checkpoint = torch.load(checkpoint_path)

losses = checkpoint['losses']
zs = checkpoint['zs']
ep_last_state_counts = checkpoint['ep_last_state_counts']
ep_last_state_trajectories = checkpoint['ep_last_state_trajectories']

# Create output file
output_path = os.path.join(os.path.dirname(checkpoint_path), "analysis_results.txt")
with open(output_path, 'w') as f:
    f.write("-" * 30 + "\n")
    f.write(f"Top {TOP_N} by visit count:\n")
    f.write("-" * 30 + "\n")

    top_count_states = sorted(
        ep_last_state_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_N]

    for state, count in top_count_states:
        trajectories = ep_last_state_trajectories[state]
        terminal_reward = trajectories[0]['rewards'][-1][0]
        
        # Calculate average trajectory reward
        traj_avgs = []
        for traj in trajectories:
            rewards = [r[0] for r in traj['rewards']]
            traj_avgs.append(sum(rewards) / len(rewards))
        avg_reward = sum(traj_avgs) / len(traj_avgs)
        
        f.write(f"State: {state}, Count: {count}, Terminal reward: {terminal_reward:.3f}, Avg trajectory reward: {avg_reward:.3f}\n")
        
        # Write each trajectory and its average reward
        for traj in trajectories:
            rewards = [r[0] for r in traj['rewards']]
            traj_avg = sum(rewards) / len(rewards)
            f.write(f"Trajectory rewards: {[f'{r:.3f}' for r in rewards]}, Average: {traj_avg:.3f}\n")
        f.write("\n")

