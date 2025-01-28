import sys
import os
import time
import datetime
import pickle
from argparse import ArgumentParser
from pathlib import Path

import numpy as np   
import torch
import matplotlib
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




# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True, help='Directory containing checkpoint file')
parser.add_argument('--trajectory_idx', type=int, default=0, help='Index of trajectory to animate') 
args = parser.parse_args()

# Load checkpoint
checkpoint_path = f"runs/{args.run_dir}/checkpoint.pt"  # checkpoint_interrupted 
checkpoint = torch.load(checkpoint_path)

losses = checkpoint['losses']
zs = checkpoint['zs']
ep_last_state_counts = checkpoint['ep_last_state_counts']
ep_last_state_trajectories = checkpoint['ep_last_state_trajectories']



# Print shape information
print("Shape of ep_last_state_trajectories:")
print(f"Number of states: {len(ep_last_state_trajectories)}")
# Get first state and its trajectories
first_state, trajectories = list(ep_last_state_trajectories.items())[0]
print(f"First state: {first_state}")
print(f"Number of trajectories for first state: {len(trajectories)}")
print(f"Example trajectory structure:")
print(f"- Number of rewards in first trajectory: {len(trajectories[0]['rewards'])}")
print(f"- Shape of rewards: {np.array(trajectories[0]['rewards']).shape}")





"""Plot and save loss curve"""
title = f"Loss and Z for the Model"
plot_loss_curve(losses, zs=zs, title=title, save_dir=os.path.dirname(checkpoint_path))
print("The final Z (partition function) estimate is {:.2f}".format(zs[-1]))






"""Show TOP_N states"""
TOP_N = 20  
REWARD_THRESHOLD = 8  
TOP_TRAJECTORIES = 11
output_path = os.path.join(os.path.dirname(checkpoint_path), "analysis_results.txt")
with open(output_path, 'w') as f:
    """By avg average trajectory rewards"""
    f.write("-" * 30 + "\n")
    f.write(f"Top {TOP_N} by Avg average trajectory rewards:\n") 
    f.write("-" * 30 + "\n")

    # Calculate average trajectory rewards for each state
    state_avg_rewards = {}
    for state, trajectories in ep_last_state_trajectories.items():
        traj_avgs = []
        for traj in trajectories:
            rewards = [r[0] for r in traj['rewards']]
            traj_avgs.append(sum(rewards) / len(rewards))
        state_avg_rewards[state] = sum(traj_avgs) / len(traj_avgs)

    top_reward_states = sorted(
        state_avg_rewards.items(),
        key=lambda x: x[1],
        reverse=True
    )[:TOP_N]

    for state, avg_reward in top_reward_states:
        trajectories = ep_last_state_trajectories[state]
        terminal_reward = trajectories[0]['rewards'][-1][0]
        count = ep_last_state_counts[state]
        
        f.write(f"State: {state}, Count: {count}, Terminal reward: {terminal_reward:.3f}, Avg trajectory reward: {avg_reward:.3f}\n")
        
        # Write each trajectory and its average reward
        for traj in trajectories:
            rewards = [r[0] for r in traj['rewards']]
            states = traj['states']
            traj_avg = sum(rewards) / len(rewards)
            f.write(f"Trajectory state-reward pairs: {[(states[i], f'${r:.3f}' if r > REWARD_THRESHOLD else f'{r:.3f}') for i,r in enumerate(rewards)]}, Average: {traj_avg:.3f}\n")
        f.write("\n")

    
    
    """Find trajectories with high rewards"""
    f.write("\n" + "-" * 30 + "\n") 
    f.write(f"Top {TOP_TRAJECTORIES} Trajectories with Rewards > {REWARD_THRESHOLD}:\n")
    f.write("-" * 30 + "\n")

    high_reward_trajectories = []
    for state, trajectories in ep_last_state_trajectories.items():
        for traj in trajectories:
            rewards = [r[0] for r in traj['rewards']]
            max_reward = max(rewards)
            if max_reward > REWARD_THRESHOLD:
                high_reward_trajectories.append((state, traj, max_reward))

    # Sort by max reward and take top N
    top_trajectories = sorted(high_reward_trajectories, key=lambda x: x[2], reverse=True)[:TOP_TRAJECTORIES]

    for i, (state, traj, max_reward) in enumerate(top_trajectories, 1):
        f.write(f"\nRank {i} - State: {state}, Max Reward: {max_reward:.3f}\n")
        rewards = [r[0] for r in traj['rewards']]
        states = traj['states']
        f.write(f"Full trajectory:\n")
        for step, (s, r) in enumerate(zip(states, rewards)):
            f.write(f"Step {step}: State={s}, Reward={r:.3f}\n")
        f.write("\n")

    


    
    

"""Create animation for the top trajectory"""
from graph.graph import draw_network_motif
import imageio


# Get the top trajectory
top_state, top_traj, _ = top_trajectories[args.trajectory_idx]
states = top_traj['states']
rewards = [r[0] for r in top_traj['rewards']]

def create_frames(states, rewards):
    frames = []
    for frame, (state, reward) in enumerate(zip(states, rewards)):
        # Create figure with two subplots side by side with width ratio 1:2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={'width_ratios': [1, 1.5]})
        fig.suptitle("Network Motif and Somite Pattern Evolution")
        
        # Draw network motif
        draw_network_motif(state, ax=ax1)
        ax1.set_title(f"Step {frame}: Network Motif")
        
        # Draw somite pattern
        somitogenesis_reward_func(state, plot=True, ax=ax2)
        ax2.set_title(f"Somite Pattern (reward: {reward:.3f})")
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)
        plt.close(fig)
        
    return frames

# Create the frames
frames = create_frames(states, rewards)

# Save as MP4 file
output_video = os.path.join(os.path.dirname(checkpoint_path), f"trajectory_evolution{args.trajectory_idx}.mp4")
imageio.mimsave(output_video, frames, fps=2)


