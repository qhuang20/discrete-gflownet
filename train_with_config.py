import sys
import os
import time
import datetime
import pickle
import numpy as np   
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from disc_gflownet.utils.setting import set_seed, set_device, tf
from disc_gflownet.utils.plotting import plot_loss_curve
from disc_gflownet.utils.logging import log_arguments, log_training_loop, track_trajectories
from disc_gflownet.utils.cache import LRUCache
from disc_gflownet.agents.tbflownet_agent import TBFlowNetAgent
from disc_gflownet.agents.dbflownet_agent import DBFlowNetAgent
from disc_gflownet.envs.grid_env import GridEnv
from disc_gflownet.envs.grid_env2 import GridEnv2
from disc_gflownet.envs.set_env import SetEnv

from scipy.integrate import solve_ivp

from threadpoolctl import threadpool_info, ThreadpoolController
from pprint import pprint

# Import environment configuration functionality
from configs import get_env, ENVS
from configs.reward_configs import get_reward_function

controller = ThreadpoolController()
controller.limit(limits=1, user_api='blas')

def compute_reward(curr_ns, env, reward_func):
    curr_ns_state = env.encoding_to_state(curr_ns)
    return reward_func(curr_ns_state) + env.min_reward

def save_checkpoint(run_dir, agent, opt, losses, zs, current_step, ep_last_state_counts, ep_last_state_trajectories, interrupted=False):
    """Save training checkpoint to file"""
    checkpoint = {
        'losses': losses,
        'zs': zs,
        'agent_state_dict': agent.model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'current_step': current_step,
        'ep_last_state_counts': ep_last_state_counts,
        'ep_last_state_trajectories': ep_last_state_trajectories,
    }
    
    filename = 'checkpoint_interrupted.pt' if interrupted else 'checkpoint.pt'
    checkpoint_path = os.path.join(run_dir, filename)
    torch.save(checkpoint, checkpoint_path)
    
    if interrupted:
        print(f"\nTraining interrupted by user.")
        print(f"Checkpoint saved to {checkpoint_path}")

def env_to_args(env_class, reward_func_name='somitogenesis'):
    """Convert an environment configuration class to an argparse-like object."""
    class Args:
        def __init__(self, env_class, reward_func_name):
            # Copy all attributes from env class
            for attr in dir(env_class):
                if not attr.startswith('_'):
                    setattr(self, attr, getattr(env_class, attr))
            
            # Set the reward function
            self.custom_reward_fn = get_reward_function(reward_func_name)
    
    return Args(env_class, reward_func_name)

def main(env_name, reward_func_name='somitogenesis'):
    global losses, zs, agent
    global ep_last_state_counts, ep_last_state_trajectories 

    # Get environment configuration and create args object
    env_class = get_env(env_name)
    args = env_to_args(env_class, reward_func_name)

    assert args.envsize == args.mbsize
    set_seed(args.seed)
    set_device(torch.device(args.device))
    
    # Environment setup based on config
    if args.env_type == 'GridEnv':
        envs = [GridEnv(args) for _ in range(args.envsize)]
    elif args.env_type == 'GridEnv2':
        envs = [GridEnv2(args) for _ in range(args.envsize)]
    else:
        raise ValueError(f"Unknown environment type: {args.env_type}")
    
    # Agent setup
    if args.method == 'tb':
        agent = TBFlowNetAgent(args, envs)
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}, {'params':[agent.log_z], 'lr': args.tb_z_lr} ])
    elif args.method == 'db' or args.method == 'fldb':
        agent = DBFlowNetAgent(args, envs)
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}])

    # Logging setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{env_name}_{args.method}_h{args.n_hid}_l{args.n_layers}_mr{args.min_reward}_ts{args.n_train_steps}_d{args.n_dims}_s{args.n_steps}_er{args.explore_ratio}_et{args.enable_time}_{reward_func_name}" 
    run_dir = os.path.join('runs', f'{timestamp}_{run_name}') 
    os.makedirs(run_dir, exist_ok=True)
    if args.log_flag:
        log_filename = os.path.join(run_dir, 'training.log') 
        log_arguments(log_filename, args)

    """Training Loop"""
    
    losses = [] 
    zs = []  
    ep_last_state_counts = {} # Counts occurrences of each episode last state.
    ep_last_state_trajectories = {}  # Store trajectories (states, actions, rewards) for each last state
    try:
        for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):            
            experiences = agent.sample_batch_episodes(args.mbsize)

            if args.n_workers > 1: 
                curr_ns_all = np.zeros((args.mbsize, args.n_steps, envs[0].encoding_dim))
                for mb in range(args.mbsize): 
                    curr_ns_all[mb] = experiences[0][mb].cpu()[1:] 
                curr_ns_all = curr_ns_all.reshape(args.mbsize*args.n_steps, envs[0].encoding_dim)

                compute_reward_partial = partial(compute_reward, env=envs[0], reward_func=args.custom_reward_fn)
                
                with Pool(processes=args.n_workers) as env_pool: 
                    curr_r_env = list(env_pool.imap(compute_reward_partial, curr_ns_all)) 
                curr_r_env = np.asarray(curr_r_env)
                curr_r_env = curr_r_env.reshape(args.mbsize, args.n_steps, 1) 

                for mb in range(args.mbsize):
                    experiences[3][mb] = torch.from_numpy(curr_r_env[mb]) 
                                
            if args.method == 'fldb':
                loss, z = agent.compute_batch_loss(experiences, use_fldb=True) 
            else:
                loss, z = agent.compute_batch_loss(experiences)     
            losses.append(loss.item())
            zs.append(z.item()) 

            loss.backward()
            opt.step()
            opt.zero_grad() 
            
            # Track trajectories 
            track_trajectories(experiences, envs[0], ep_last_state_counts, ep_last_state_trajectories, i) 
            if i % args.log_freq == 0 and args.log_flag:
                log_training_loop(log_filename, agent, i, ep_last_state_counts, ep_last_state_trajectories) 

            # Save checkpoint every log_freq episodes
            if i % args.log_freq == 0:
                save_checkpoint(run_dir, agent, opt, losses, zs, i, ep_last_state_counts, ep_last_state_trajectories)
                
    except KeyboardInterrupt:
        save_checkpoint(run_dir, agent, opt, losses, zs, i, ep_last_state_counts, ep_last_state_trajectories, interrupted=True)
        sys.exit(0)  # Gracefully exit the program
    except Exception as e:
        # Handle other exceptions
        print("\nAn unexpected error occurred:", e)
        raise

    return losses, zs, agent, ep_last_state_counts, ep_last_state_trajectories 

if __name__ == '__main__':
    print(f"Available OS CPU threads: {os.cpu_count()}")
    print(f"Default PyTorch CPU threads: {torch.get_num_threads()}")
    
    argparser = ArgumentParser(description='GFlowNet for Genetic Circuits Design.') 
    
    # Environment selection
    argparser.add_argument('--env', type=str, default='gridenv', 
                          help=f'Environment to use. Available: {list(ENVS.keys())}')
    argparser.add_argument('--reward', type=str, default='somitogenesis',
                          help='Reward function to use. Available: coord, oscillator, somitogenesis')
    
    args = argparser.parse_args()
    
    # Print environment configuration info
    print(f"Using environment: {args.env}")
    print(f"Using reward function: {args.reward}")
    print("To modify parameters, edit the environment config files directly in configs/")
    
    main(args.env, args.reward) 

