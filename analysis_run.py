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




# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True, help='Directory containing checkpoint file')
parser.add_argument('--trajectory_idx', type=int, default=0, help='Index of trajectory to animate')
parser.add_argument('--reward_threshold', type=float, default=0.1, help='Reward threshold for mode counting')
parser.add_argument('--similarity_threshold', type=float, default=0.99, help='Similarity threshold for mode counting')
parser.add_argument('--top_k', type=int, default=10, help='Number of top states to track')
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
print(f"Number of ep_last_state: {len(ep_last_state_trajectories)}")
# Get first episode's last state and its trajectories
first_ep_last_state, trajectories = list(ep_last_state_trajectories.items())[0]
print(f"First episode's last state: {first_ep_last_state}")
print(f"Number of trajectories for first episode's last state: {len(trajectories)}")
print(f"Example trajectory structure:")
print(f"- Number of rewards in first trajectory: {len(trajectories[0]['rewards'])}")
print(f"- Shape of rewards: {np.array(trajectories[0]['rewards']).shape}")




"""Plot and save loss curve"""
title = f"Loss and Z for the Model"
plot_loss_curve(losses, zs=zs, title=title, save_dir=os.path.dirname(checkpoint_path))
print("The final Z (partition function) estimate is {:.2f}".format(zs[-1]))







"""Mode counting and top rewards tracking - Optimized Version with cKDTree"""
start_time = time.time()

sorted_trajectories = []  # sort by training step
for last_state, trajectories in ep_last_state_trajectories.items():
    for traj in trajectories:
        sorted_trajectories.append((traj['training_step'], last_state, traj))
sorted_trajectories.sort(key=lambda x: x[0])

# Initialize data structures for mode tracking
modes_dict = {}  # Dictionary to store modes and their info
mode_list = []   # List to maintain order of discovery
top_modes_avg_rewards = []
tree = None  # Initialize cKDTree as None

# Process trajectories in batches for efficiency
BATCH_SIZE = 1024
for i in range(0, len(sorted_trajectories), BATCH_SIZE):
    batch = sorted_trajectories[i:i + BATCH_SIZE]
    
    # Process each trajectory in the batch
    for training_step, last_state, traj in batch:
        rewards = np.array([r[0] for r in traj['rewards']])
        states = np.array(traj['states'])
        
        # Filter states by reward threshold
        mask = rewards > args.reward_threshold
        high_reward_states = states[mask]
        high_rewards = rewards[mask]
        
        if len(high_reward_states) == 0:
            continue
            
        # Process each high reward state
        for state, reward in zip(high_reward_states, high_rewards):
            state_tuple = tuple(state)
            
            if state_tuple in modes_dict:
                continue
                
            if len(modes_dict) == 0:
                modes_dict[state_tuple] = {
                    'reward': reward,
                    'step': training_step
                }
                mode_list.append(state_tuple)
                # Add initial top-k average reward
                rewards_list = [modes_dict[m]['reward'] for m in mode_list]
                top_k_rewards = sorted(rewards_list, reverse=True)[:min(len(mode_list), args.top_k)]
                top_modes_avg_rewards.append(sum(top_k_rewards) / args.top_k)
                continue
            
            # Use cKDTree for efficient similarity search when modes > 1.. TODO
            if len(modes_dict) > 1: 
                if tree is None:
                    # Initialize tree with existing modes
                    existing_modes = np.array([list(m) for m in mode_list])
                    tree = cKDTree(existing_modes)
                    
                # Query the tree for nearest neighbors
                distances, _ = tree.query(state.reshape(1, -1), k=1)
                similarity = np.exp(-distances[0])  # Converts euclidean distance into a decaying similarity score [0,1]
                
                if similarity <= args.similarity_threshold:
                    modes_dict[state_tuple] = {
                        'reward': reward,
                        'step': training_step
                    }
                    mode_list.append(state_tuple)
                    # Rebuild tree with new mode
                    existing_modes = np.array([list(m) for m in mode_list])
                    tree = cKDTree(existing_modes)
                    
                    # Update top-k average rewards
                    rewards_list = [modes_dict[m]['reward'] for m in mode_list]
                    top_k_rewards = sorted(rewards_list, reverse=True)[:min(len(mode_list), args.top_k)]
                    top_modes_avg_rewards.append(sum(top_k_rewards) / args.top_k)
            else:
                # Use original cosine similarity for smaller number of modes
                state_array = np.array(state).reshape(1, -1)
                existing_modes = np.array([list(m) for m in mode_list])
                similarities = cosine_similarity(state_array, existing_modes)[0]
                
                if not np.any(similarities > args.similarity_threshold):
                    modes_dict[state_tuple] = {
                        'reward': reward,
                        'step': training_step
                    }
                    mode_list.append(state_tuple)
                    # Update top-k average rewards
                    rewards_list = [modes_dict[m]['reward'] for m in mode_list]
                    top_k_rewards = sorted(rewards_list, reverse=True)[:min(len(mode_list), args.top_k)]
                    top_modes_avg_rewards.append(sum(top_k_rewards) / args.top_k)

end_time = time.time()
print(f"\nMode counting took {end_time - start_time:.2f} seconds")
print(f"Found {len(modes_dict)} distinct modes with reward threshold {args.reward_threshold} "
      f"and similarity threshold {args.similarity_threshold}")

# Plot mode discovery and top-k average rewards
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

discovery_steps = [modes_dict[m]['step'] for m in mode_list]

# Mode discovery plot
ax1.plot(discovery_steps, range(1, len(modes_dict) + 1), '-o')
ax1.set_xlabel('Training Step')
ax1.set_ylabel('Number of Modes Found')
ax1.set_title('Mode Discovery Progress')
ax1.grid(True)

# Top-k average rewards plot
ax2.plot(discovery_steps, top_modes_avg_rewards, '-o')
ax2.set_xlabel('Training Step')
ax2.set_ylabel(f'Average Reward of Top-{args.top_k} Modes')
ax2.set_title(f'Top-{args.top_k} Modes Average Reward Progress')
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(checkpoint_path), 'mode_and_rewards_discovery.png'))
plt.close()






"""Save modes information to file"""
output_path = os.path.join(os.path.dirname(checkpoint_path), "analysis_results.txt")
final_top_k = sorted(
    [(mode, info['reward']) for mode, info in modes_dict.items()],
    key=lambda x: x[1],
    reverse=True
)[:args.top_k]

with open(output_path, 'w') as f:
    f.write("-" * 30 + "\n")
    f.write(f"Discovered Modes Information:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Found {len(modes_dict)} distinct modes with:\n")
    f.write(f"- Reward threshold: {args.reward_threshold}\n")
    f.write(f"- Similarity threshold: {args.similarity_threshold}\n\n")
    
    for i, mode in enumerate(mode_list, 1):
        info = modes_dict[mode]
        f.write(f"Mode {i}:\n")
        f.write(f"- State: {list(mode)}\n")
        f.write(f"- Discovered at step: {info['step']}\n")
    f.write("\n")
    
    f.write("-" * 30 + "\n")
    f.write(f"Final Top-{args.top_k} Modes by Reward:\n")
    f.write("-" * 30 + "\n")
    for i, (state, reward) in enumerate(final_top_k, 1):
        f.write(f"Rank {i}:\n")
        f.write(f"- State: {list(state)}\n")
        f.write(f"- Reward: {reward:.3f}\n")
    f.write("\n")








# """Save top performing final states and their trajectories to file"""
# TOP_N = 20  
# TOP_REWARD_THRESHOLD = 8  
# TOP_TRAJECTORIES = 11

# with open(output_path, 'a') as f:
#     """By avg average trajectory rewards"""
#     f.write("-" * 30 + "\n")
#     f.write(f"Top {TOP_N} Final States by Avg Average Trajectory Rewards:\n") 
#     f.write("-" * 30 + "\n")

#     # Calculate avg average trajectory rewards for each final state
#     final_state_avg_rewards = {}
#     for final_state, trajectories in ep_last_state_trajectories.items():
#         traj_avgs = []
#         for traj in trajectories:
#             rewards = [r[0] for r in traj['rewards']]
#             traj_avgs.append(sum(rewards) / len(rewards))
#         final_state_avg_rewards[final_state] = sum(traj_avgs) / len(traj_avgs)

#     top_reward_final_states = sorted(
#         final_state_avg_rewards.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )[:TOP_N]

#     for final_state, avg_reward in top_reward_final_states:
#         trajectories = ep_last_state_trajectories[final_state]
#         terminal_reward = trajectories[0]['rewards'][-1][0]
#         count = ep_last_state_counts[final_state]
        
#         f.write(f"Final State: {final_state}, Count: {count}, Terminal Reward: {terminal_reward:.3f}, Avg Average Trajectory Reward: {avg_reward:.3f}\n")
        
#         # Write each trajectory and its average reward
#         for traj in trajectories:
#             rewards = [r[0] for r in traj['rewards']]
#             states = traj['states']
#             traj_avg = sum(rewards) / len(rewards)
#             f.write(f"Trajectory state-reward pairs: {[(states[i], f'${r:.3f}' if r > TOP_REWARD_THRESHOLD else f'{r:.3f}') for i,r in enumerate(rewards)]}, Average: {traj_avg:.3f}\n")
#         f.write("\n")

#     """By final state's visit count"""
#     f.write("\n" + "-" * 30 + "\n")
#     f.write(f"Top {TOP_N} Final States by Visit Count:\n")
#     f.write("-" * 30 + "\n")

#     top_count_final_states = sorted(
#         ep_last_state_counts.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )[:TOP_N]

#     for final_state, count in top_count_final_states:
#         trajectories = ep_last_state_trajectories[final_state]
#         terminal_reward = trajectories[0]['rewards'][-1][0]
#         avg_reward = final_state_avg_rewards[final_state]
        
#         f.write(f"Final State: {final_state}, Visit Count: {count}, Terminal Reward: {terminal_reward:.3f}, Avg Average Trajectory Reward: {avg_reward:.3f}\n")
        
#         # Write each trajectory and its average reward
#         for traj in trajectories:
#             rewards = [r[0] for r in traj['rewards']]
#             states = traj['states']
#             traj_avg = sum(rewards) / len(rewards)
#             f.write(f"Trajectory state-reward pairs: {[(states[i], f'{r:.3f}') for i,r in enumerate(rewards)]}, Average: {traj_avg:.3f}\n")
#         f.write("\n")

#     """Find trajectories with high rewards leading to final states"""
#     f.write("\n" + "-" * 30 + "\n") 
#     f.write(f"Top {TOP_TRAJECTORIES} Trajectories with Rewards > {TOP_REWARD_THRESHOLD}:\n")
#     f.write("-" * 30 + "\n")

#     high_reward_trajectories = []
#     for final_state, trajectories in ep_last_state_trajectories.items():
#         for traj in trajectories:
#             rewards = [r[0] for r in traj['rewards']]
#             max_reward = max(rewards)
#             if max_reward > TOP_REWARD_THRESHOLD:
#                 high_reward_trajectories.append((final_state, traj, max_reward))

#     # Sort by max reward and take top N
#     top_trajectories = sorted(high_reward_trajectories, key=lambda x: x[2], reverse=True)[:TOP_TRAJECTORIES]

#     for i, (final_state, traj, max_reward) in enumerate(top_trajectories, 1):
#         f.write(f"\nRank {i} - Final State: {final_state}, Max Reward: {max_reward:.3f}\n")
#         rewards = [r[0] for r in traj['rewards']]
#         states = traj['states']
#         f.write(f"Full trajectory:\n")
#         for step, (s, r) in enumerate(zip(states, rewards)):
#             f.write(f"Step {step}: State={s}, Reward={r:.3f}\n")
#         f.write("\n")


    









# """Create animation for the top trajectory"""
# from graph.graph import draw_network_motif
# import imageio


# # Get the top trajectory
# top_state, top_traj, _ = top_trajectories[args.trajectory_idx]
# states = top_traj['states']
# rewards = [r[0] for r in top_traj['rewards']]

# def create_frames(states, rewards):
#     frames = []
#     for frame, (state, reward) in enumerate(zip(states, rewards)):
#         # Create figure with two subplots side by side with width ratio 1:2
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={'width_ratios': [1, 1.5]})
#         fig.suptitle("Network Motif and Somite Pattern Evolution")
        
#         # Draw network motif
#         draw_network_motif(state, ax=ax1)
#         ax1.set_title(f"Step {frame}: Network Motif")
        
#         # Draw somite pattern
#         somitogenesis_reward_func(state, plot=True, ax=ax2)
#         ax2.set_title(f"Somite Pattern (reward: {reward:.3f})")
        
#         plt.tight_layout()
        
#         # Convert plot to image
#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         frames.append(image)
#         plt.close(fig)
        
#     return frames

# # Create the frames
# frames = create_frames(states, rewards)

# # Save as MP4 file
# output_video = os.path.join(os.path.dirname(checkpoint_path), f"trajectory_evolution{args.trajectory_idx}.mp4")
# imageio.mimsave(output_video, frames, fps=2)




