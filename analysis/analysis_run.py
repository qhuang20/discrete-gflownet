import sys
import os
import time
import datetime
import pickle
from argparse import ArgumentParser
import heapq

import numpy as np   
import torch
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from scipy.spatial import cKDTree

import sys
import os

# Add the parent directory to the path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from disc_gflownet.utils.plotting import plot_loss_curve
from graph.graph import plot_network_motifs_and_somites
from analysis.helper_diversity_selection import select_diverse_modes


# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True, help='Directory containing checkpoint file')
# parser.add_argument('--trajectory_idx', type=int, default=0, help='Index of trajectory to animate')
parser.add_argument('--reward_threshold', type=float, default=0.1, help='Reward threshold for mode counting')
parser.add_argument('--top_k', type=int, default=10, help='Number of top states to track')
parser.add_argument('--top_m', type=int, default=24, help='Number of top states to plot') 
parser.add_argument('--diversity_metric', type=str, default='combined', 
                    choices=['structure', 'parameters', 'rewards', 'topology', 'combined'], 
                    help='Diversity metric to use for selecting diverse modes')
parser.add_argument('--n_diverse', type=int, default=24, help='Number of diverse modes to plot')
parser.add_argument('--reward_range', type=float, nargs=2, default=None, 
                    metavar=('MIN', 'MAX'), help='Reward range [min, max] to plot modes for (e.g., 20 30)')
parser.add_argument('--skip_mode_discovery', action='store_true', 
                    help='Skip mode discovery and use cached results (for faster reward range analysis)')
parser.add_argument('--skip_plots', action='store_true', 
                    help='Skip expensive plotting operations (for faster analysis)')
args = parser.parse_args()

# Load checkpoint
checkpoint_path = f"runs/{args.run_dir}/checkpoint.pt"  # checkpoint_interrupted 
checkpoint = torch.load(checkpoint_path, weights_only=False)
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







"""PLOT: Plot and save loss curve"""
title = f"Loss and Z for the Model"
plot_loss_curve(losses, zs=zs, title=title, save_dir=os.path.dirname(checkpoint_path))
print("The final Z (partition function) estimate is {:.2f}".format(zs[-1]))






"""PLOT: Mode counting and top rewards tracking (--reward_threshold) """
# Check if we can load cached results
modes_save_path = os.path.join(os.path.dirname(checkpoint_path), "modes_with_trajectories.pkl")
# cached_results_path = os.path.join(os.path.dirname(checkpoint_path), "cached_analysis_results.pkl")

if args.skip_mode_discovery and os.path.exists(modes_save_path):
    print("Loading cached mode discovery results...")
    with open(modes_save_path, 'rb') as f:
        modes_dict = pickle.load(f)
    print(f"Loaded {len(modes_dict)} cached modes")
    
    # Reconstruct mode_list and top_modes_avg_rewards from cached data
    mode_list = list(modes_dict.keys())
    all_rewards = [info['reward'] for info in modes_dict.values()]
    top_k_rewards = sorted(all_rewards, reverse=True)[:args.top_k]
    top_modes_avg_rewards = []
    for i in range(len(mode_list)):
        current_top_k = sorted(all_rewards[:i+1], reverse=True)[:args.top_k]
        top_modes_avg_rewards.append(sum(current_top_k) / len(current_top_k))
    
else:
    start_time = time.time()

    # Convert trajectories to numpy arrays for faster processing
    sorted_trajectories = [(traj['training_step'], last_state, traj) 
                          for last_state, trajectories in ep_last_state_trajectories.items()
                          for traj in trajectories]
    sorted_trajectories.sort(key=lambda x: x[0])
    print("\nFirst 10 sorted trajectories:")
    for i, (step, last_state, traj) in enumerate(sorted_trajectories[:10]):
        print(f"{i}. Step: {step}, Last state: {last_state}")

    modes_dict = {}  # Dictionary to store modes and their info
    modes_set = set()  # Set for faster membership testing
    mode_list = []   # List to maintain order of discovery
    top_modes_avg_rewards = []
    top_k_rewards = []  # Min-heap to store top-k rewards

    BATCH_SIZE = 512  # Process trajectories in batches for efficiency
    for i in range(0, len(sorted_trajectories), BATCH_SIZE):
        batch = sorted_trajectories[i:i + BATCH_SIZE]
        
        # Vectorized processing of batch
        batch_rewards = np.array([np.array([r[0] for r in traj[2]['rewards']]) for traj in batch])
        batch_states = np.array([np.array(traj[2]['states']) for traj in batch])
        batch_steps = np.array([traj[0] for traj in batch])
        
        # Filter states by reward threshold using vectorized operations
        mask = batch_rewards > args.reward_threshold
        
        for b_idx in range(len(batch)):
            high_reward_states = batch_states[b_idx][mask[b_idx]]
            high_rewards = batch_rewards[b_idx][mask[b_idx]]
            training_step = batch_steps[b_idx]
            
            if len(high_reward_states) == 0:
                continue
                
            # Process high reward states
            for state, reward in zip(high_reward_states, high_rewards):
                state_tuple = tuple(state)
                
                if state_tuple in modes_set:
                    continue
                    
                modes_set.add(state_tuple)
                # Store full trajectory info for this mode
                trajectory = batch[b_idx][2]
                modes_dict[state_tuple] = {
                    'reward': reward,
                    'step': training_step,
                    'states': trajectory['states'],
                    'rewards': [r[0] for r in trajectory['rewards']]
                }
                mode_list.append(state_tuple)
                
                # Efficient rolling top-k update using min-heap
                if len(top_k_rewards) < args.top_k:
                    heapq.heappush(top_k_rewards, reward)
                elif reward > top_k_rewards[0]:
                    heapq.heapreplace(top_k_rewards, reward)
                    
                # top_modes_avg_rewards.append(np.mean(top_k_rewards))
                top_modes_avg_rewards.append(sum(top_k_rewards) / args.top_k)

    end_time = time.time()
    print(f"\nMode counting took {end_time - start_time:.2f} seconds")
    print(f"Found {len(modes_dict)} distinct modes with reward threshold {args.reward_threshold}\n\n")

    # Save all modes and their trajectories
    with open(modes_save_path, 'wb') as f:
        pickle.dump(modes_dict, f)
    print(f"\nSaved all modes and their trajectories to: {modes_save_path}\n")

# Plot mode discovery and top-k average rewards
if not args.skip_plots:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    discovery_steps = np.array([modes_dict[m]['step'] for m in mode_list])

    # Mode discovery plot
    ax1.plot(discovery_steps, np.arange(1, len(modes_dict) + 1), '-o')
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







"""PLOT: Save modes information to file and Plot top_m (--top_m)"""
# file output_path = analysis_results.txt 
output_path = os.path.join(os.path.dirname(checkpoint_path), "analysis_results.txt")
top_m = sorted(
    [(mode, info['reward']) for mode, info in modes_dict.items()],
    key=lambda x: x[1],
    reverse=True
)[:args.top_m]

with open(output_path, 'w') as f:
    f.write("-" * 30 + "\n")
    f.write(f"Discovered Modes Information:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Found {len(modes_dict)} distinct modes with:\n")
    f.write(f"- Reward threshold: {args.reward_threshold}\n\n")
    
    for i, mode in enumerate(mode_list, 1):
        info = modes_dict[mode]
        f.write(f"Mode {i}:\n")
        f.write(f"- State: {list(mode)}\n")
        f.write(f"- Reward: {info['reward']:.3f}\n")
        f.write(f"- Top-{args.top_k} modes avg reward so far: {top_modes_avg_rewards[i-1]:.3f}\n")
        f.write(f"- Discovered at step: {info['step']}\n")
    f.write("\n")
    
    f.write("-" * 30 + "\n")
    f.write(f"Top-{args.top_m} Modes by Reward:\n")
    f.write("-" * 30 + "\n")
    for i, (state, reward) in enumerate(top_m, 1):
        f.write(f"Rank {i}:\n")
        f.write(f"- State: {list(state)}\n")
        f.write(f"- Reward: {reward:.3f}\n")
    f.write("\n")

# Plot network motifs and somite patterns for top modes
if not args.skip_plots:
    top_m_states = [list(state) for state, _ in top_m]
    motifs_plot_path = os.path.join(os.path.dirname(checkpoint_path), "top_modes_motifs_and_somites.png")
    plot_network_motifs_and_somites(top_m_states, save_path=motifs_plot_path)
    print(f"\nNetwork motifs and somite patterns saved to: {motifs_plot_path}")





"""PLOT: Plot diverse modes (--diversity_metric --n_diverse)"""
if not args.skip_plots:
    diverse_states = select_diverse_modes(modes_dict, n_diverse=args.n_diverse, diversity_metric=args.diversity_metric)
    diverse_plot_path = os.path.join(os.path.dirname(checkpoint_path), f"diverse_modes_{args.diversity_metric}_motifs_and_somites.png")
    plot_network_motifs_and_somites(diverse_states, save_path=diverse_plot_path)
    print(f"Diverse modes ({args.diversity_metric} metric) motifs and somite patterns saved to: {diverse_plot_path}")
else:
    diverse_states = select_diverse_modes(modes_dict, n_diverse=args.n_diverse, diversity_metric=args.diversity_metric)

# Add diverse modes information to the output file
with open(output_path, 'a') as f:
    f.write("-" * 30 + "\n")
    f.write(f"Diverse Modes ({args.diversity_metric.capitalize()} Diversity Metric):\n")
    f.write("-" * 30 + "\n")
    for i, state in enumerate(diverse_states, 1):
        state_tuple = tuple(state)
        if state_tuple in modes_dict:
            info = modes_dict[state_tuple]
            f.write(f"Diverse Mode {i}:\n")
            f.write(f"- State: {state}\n")
            f.write(f"- Reward: {info['reward']:.3f}\n")
            f.write(f"- Discovered at step: {info['step']}\n")
        f.write("\n")





"""PLOT: Reward distribution (reward vs count)"""
# Extract all rewards from modes
all_rewards = [info['reward'] for info in modes_dict.values()]

if not args.skip_plots:
    # Create histogram of rewards
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Histogram
    ax1.hist(all_rewards, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Count')
    ax1.set_title('Reward Distribution (Histogram)')
    ax1.grid(True, alpha=0.3)

    # Log-scale histogram for better visualization of tail
    ax2.hist(all_rewards, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_yscale('log')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('Reward Distribution (Log Scale)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    reward_dist_path = os.path.join(os.path.dirname(checkpoint_path), 'reward_distribution.png')
    plt.savefig(reward_dist_path)
    plt.close()

    print(f"Reward distribution plot saved to: {reward_dist_path}")

# Add reward distribution statistics to the output file
with open(output_path, 'a') as f:
    f.write("-" * 30 + "\n")
    f.write(f"Reward Distribution Statistics:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total modes: {len(all_rewards)}\n")
    f.write(f"Mean reward: {np.mean(all_rewards):.3f}\n")
    f.write(f"Median reward: {np.median(all_rewards):.3f}\n")
    f.write(f"Std reward: {np.std(all_rewards):.3f}\n")
    f.write(f"Min reward: {np.min(all_rewards):.3f}\n")
    f.write(f"Max reward: {np.max(all_rewards):.3f}\n")
    f.write(f"25th percentile: {np.percentile(all_rewards, 25):.3f}\n")
    f.write(f"75th percentile: {np.percentile(all_rewards, 75):.3f}\n")
    f.write(f"95th percentile: {np.percentile(all_rewards, 95):.3f}\n")
    f.write(f"99th percentile: {np.percentile(all_rewards, 99):.3f}\n\n")





"""PLOT: Modes within specific reward range (--reward_range)"""
if args.reward_range is not None:
    min_reward, max_reward = args.reward_range
    
    # Filter modes within the specified reward range
    range_modes = []
    for state_tuple, info in modes_dict.items():
        if min_reward <= info['reward'] <= max_reward:
            range_modes.append(list(state_tuple))
    
    if range_modes:
        print(f"\nFound {len(range_modes)} modes with rewards in range [{min_reward}, {max_reward}]")
        
        # Plot network motifs and somite patterns for modes in range
        # Always plot if reward_range is specified, regardless of skip_plots flag
        range_plot_path = os.path.join(os.path.dirname(checkpoint_path), f"modes_reward_range_{min_reward}_{max_reward}_motifs_and_somites.png")
        plot_network_motifs_and_somites(range_modes, save_path=range_plot_path)
        print(f"Modes in reward range [{min_reward}, {max_reward}] saved to: {range_plot_path}")
        
        # Add range modes information to the output file
        with open(output_path, 'a') as f:
            f.write("-" * 30 + "\n")
            f.write(f"Modes in Reward Range [{min_reward}, {max_reward}]:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Found {len(range_modes)} modes in this range\n\n")
            
            # Sort by reward for better organization
            range_modes_with_rewards = []
            for state in range_modes:
                state_tuple = tuple(state)
                if state_tuple in modes_dict:
                    range_modes_with_rewards.append((state, modes_dict[state_tuple]['reward']))
            
            range_modes_with_rewards.sort(key=lambda x: x[1], reverse=True)
            
            for i, (state, reward) in enumerate(range_modes_with_rewards, 1):
                f.write(f"Mode {i}:\n")
                f.write(f"- State: {state}\n")
                f.write(f"- Reward: {reward:.3f}\n")
                f.write(f"- Discovered at step: {modes_dict[tuple(state)]['step']}\n\n")
    else:
        print(f"\nNo modes found with rewards in range [{min_reward}, {max_reward}]")
        with open(output_path, 'a') as f:
            f.write("-" * 30 + "\n")
            f.write(f"Modes in Reward Range [{min_reward}, {max_reward}]:\n")
            f.write("-" * 30 + "\n")
            f.write(f"No modes found in this range\n\n")













# """Save top performing final states and their trajectories to file"""
# TOP_N = 20  
# TOP_REWARD_THRESHOLD = 8  
# TOP_TRAJECTORIES = 11

# with open(output_path, 'a') as f:
#     """By avg average trajectory rewards"""
#     f.write("-" * 30 + "\n")
#     f.write(f"Top {TOP_N} Final States by Avg Average Trajectory Rewards:\n") 
#     f.write("-" * 30 + "\n")

#     # Calculate avg average trajectory rewards for each final state (but in somite, we are likely to only have one traj)
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
#             f.write("Trajectory state-reward pairs:\n")
#             for i, (state, reward) in enumerate(zip(states, rewards)):
#                 reward_str = f'${reward:.3f}' if reward > TOP_REWARD_THRESHOLD else f'{reward:.3f}'
#                 f.write(f"  Step {i}: State={state}, Reward={reward_str}\n")
#             f.write(f"Average: {traj_avg:.3f}\n\n")

    # """By final state's visit count"""
    # f.write("\n" + "-" * 30 + "\n")
    # f.write(f"Top {TOP_N} Final States by Visit Count:\n")
    # f.write("-" * 30 + "\n")

    # top_count_final_states = sorted(
    #     ep_last_state_counts.items(),
    #     key=lambda x: x[1],
    #     reverse=True
    # )[:TOP_N]

    # for final_state, count in top_count_final_states:
    #     trajectories = ep_last_state_trajectories[final_state]
    #     terminal_reward = trajectories[0]['rewards'][-1][0]
    #     avg_reward = final_state_avg_rewards[final_state]
        
    #     f.write(f"Final State: {final_state}, Visit Count: {count}, Terminal Reward: {terminal_reward:.3f}, Avg Average Trajectory Reward: {avg_reward:.3f}\n")
        
    #     # Write each trajectory and its average reward
    #     for traj in trajectories:
    #         rewards = [r[0] for r in traj['rewards']]
    #         states = traj['states']
    #         traj_avg = sum(rewards) / len(rewards)
    #         f.write(f"Trajectory state-reward pairs: {[(states[i], f'{r:.3f}') for i,r in enumerate(rewards)]}, Average: {traj_avg:.3f}\n")
    #     f.write("\n")

    # """Find trajectories with high rewards leading to final states"""
    # f.write("\n" + "-" * 30 + "\n") 
    # f.write(f"Top {TOP_TRAJECTORIES} Trajectories with Rewards > {TOP_REWARD_THRESHOLD}:\n")
    # f.write("-" * 30 + "\n")

    # high_reward_trajectories = []
    # for final_state, trajectories in ep_last_state_trajectories.items():
    #     for traj in trajectories:
    #         rewards = [r[0] for r in traj['rewards']]
    #         max_reward = max(rewards)
    #         if max_reward > TOP_REWARD_THRESHOLD:
    #             high_reward_trajectories.append((final_state, traj, max_reward))

    # # Sort by max reward and take top N
    # top_trajectories = sorted(high_reward_trajectories, key=lambda x: x[2], reverse=True)[:TOP_TRAJECTORIES]

    # for i, (final_state, traj, max_reward) in enumerate(top_trajectories, 1):
    #     f.write(f"\nRank {i} - Final State: {final_state}, Max Reward: {max_reward:.3f}\n")
    #     rewards = [r[0] for r in traj['rewards']]
    #     states = traj['states']
    #     f.write(f"Full trajectory:\n")
    #     for step, (s, r) in enumerate(zip(states, rewards)):
    #         f.write(f"Step {step}: State={s}, Reward={r:.3f}\n")
    #     f.write("\n")


    































# """Create animation for the top trajectories by average reward"""
# from graph.graph import draw_network_motif
# import imageio

# # Get trajectories sorted by avg average trajectory rewards
# top_reward_final_states = sorted(
#     final_state_avg_rewards.items(),
#     key=lambda x: x[1], 
#     reverse=True
# )

# # # Get the specified trajectory
# # final_state = top_reward_final_states[args.trajectory_idx][0]
# # trajectories = ep_last_state_trajectories[final_state]
# # traj = trajectories[0]  # Take first trajectory for this final state
# # states = traj['states']
# # rewards = [r[0] for r in traj['rewards']]

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


# # # Create the frames
# # frames = create_frames(states, rewards)

# # # Save as MP4 file
# # output_video = os.path.join(os.path.dirname(checkpoint_path), f"trajectory_evolution{args.trajectory_idx}.mp4")
# # imageio.mimsave(output_video, frames, fps=2)







# # Create animations for top 5 trajectories
# for idx in range(min(5, len(top_reward_final_states))):
#     final_state = top_reward_final_states[idx][0]
#     trajectories = ep_last_state_trajectories[final_state]
#     traj = trajectories[0]  # Take first trajectory for this final state
#     states = traj['states']
#     rewards = [r[0] for r in traj['rewards']]
    
#     print(f"Creating animation for trajectory {idx+1} of {min(5, len(top_reward_final_states))}")
    
#     # Create the frames
#     frames = create_frames(states, rewards)
    
#     # Save as MP4 file
#     output_video = os.path.join(os.path.dirname(checkpoint_path), f"trajectory_evolution_{idx+1}.mp4")
#     imageio.mimsave(output_video, frames, fps=2)
#     print(f"Saved animation to: {output_video}")

# print("All animations completed!")






