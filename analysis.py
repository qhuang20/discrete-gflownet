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




# Load checkpoint
run_dir = "20250118_173924_fldb_h256_l3_mr0.001_ts2000_d9_s71_er0.35_etFalse"
checkpoint_path = f"runs/{run_dir}/checkpoint_interrupted.pt"
checkpoint = torch.load(checkpoint_path)

losses = checkpoint['losses']
zs = checkpoint['zs']
ep_last_state_counts = checkpoint['ep_last_state_counts']
ep_last_state_trajectories = checkpoint['ep_last_state_trajectories']



# Print shape information
print("Shape of ep_last_state_trajectories:")
print(f"Number of states: {len(ep_last_state_trajectories)}")
for state, trajectories in list(ep_last_state_trajectories.items())[:1]:  # Look at first state
    print(f"Number of trajectories per state: {len(trajectories)}")
    print(f"Example trajectory structure:")
    print(f"- Number of rewards in first trajectory: {len(trajectories[0]['rewards'])}")
    print(f"- Shape of rewards: {np.array(trajectories[0]['rewards']).shape}")
    
    


import matplotlib.pyplot as plt

# Example loss list (replace with your actual data)
loss_list = [0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_list, label='Training Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

# Save the plot to a file


# Show the plot
# plt.show()





"""Plot and save loss curve"""

title = f"Loss and Z for the Model"
plot_loss_curve(losses, zs=zs, title=title, save_dir=os.path.dirname(checkpoint_path))
print("The final Z (partition function) estimate is {:.2f}".format(zs[-1]))






"""Show TOP_N states"""

TOP_N = 20  # Number of top states to analyze
output_path = os.path.join(os.path.dirname(checkpoint_path), "analysis_results.txt")
with open(output_path, 'w') as f:
    f.write("-" * 30 + "\n")
    f.write(f"Top {TOP_N} by Avg trajectory rewards:\n")
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
            f.write(f"Trajectory state-reward pairs: {[(states[i], f'{r:.3f}') for i,r in enumerate(rewards)]}, Average: {traj_avg:.3f}\n")
        f.write("\n")




