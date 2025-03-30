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
from reward_func.evo_devo import coord_reward_func, oscillator_reward_func, somitogenesis_reward_func

from threadpoolctl import threadpool_info, ThreadpoolController
from pprint import pprint
controller = ThreadpoolController()
controller.limit(limits=1, user_api='blas')
# pprint(threadpool_info())
# exit()





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







def main(args):
    global losses, zs, agent
    global ep_last_state_counts, ep_last_state_trajectories 

    assert args.envsize == args.mbsize
    set_seed(args.seed)
    set_device(torch.device(args.device))
    
    # Environment setup 
    # envs = [GridEnv(args) for _ in range(args.envsize)]
    envs = [GridEnv2(args) for _ in range(args.envsize)]
    
    # Agent setup
    if args.method == 'tb':
        agent = TBFlowNetAgent(args, envs)
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}, {'params':[agent.log_z], 'lr': args.tb_z_lr} ])
    elif args.method == 'db' or args.method == 'fldb':
        agent = DBFlowNetAgent(args, envs)
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}])

    # Logging setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.method}_h{args.n_hid}_l{args.n_layers}_mr{args.min_reward}_ts{args.n_train_steps}_d{args.n_dims}_s{args.n_steps}_er{args.explore_ratio}_et{args.enable_time}" 
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
                # print("Start Multiprocessing !") 
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
                # print("Multiprocessing done !") 
            
                                
                
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

            # Save checkpoint every 1000 episodes
            if i % 1000 == 0:
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
    # torch.set_num_threads(1) 
    
    argparser = ArgumentParser(description='GFlowNet for Genetic Circuits Design.')
    
    # Training
    argparser.add_argument('--device', type=str, default='cpu') # cuda
    argparser.add_argument('--progress', type=bool, default=True)
    argparser.add_argument('--seed', type=int, default=42) 
    # argparser.add_argument('--n_train_steps', type=int, default=1000) 
    argparser.add_argument('--n_train_steps', type=int, default=2000) 
    argparser.add_argument('--n_workers', type=int, default=10) 
    argparser.add_argument('--cache_max_size', type=int, default=10_000) # cache will be used when n_workers == 1 
    # argparser.add_argument('--log_freq', type=int, default=100) 
    argparser.add_argument('--log_freq', type=int, default=1000) 
    argparser.add_argument('--log_flag', type=bool, default=True)
    argparser.add_argument('--mbsize', type=int, default=8) 
    
    # Model 
    # argparser.add_argument('--method', type=str, default='tb') 
    argparser.add_argument('--method', type=str, default='fldb') 
    # argparser.add_argument('--explore_ratio', type=float, default=0.06) 
    argparser.add_argument('--explore_ratio', type=float, default=0.05) 
    argparser.add_argument('--learning_rate', type=float, default=1e-3)
    argparser.add_argument('--tb_lr', type=float, default=0.01)
    argparser.add_argument('--tb_z_lr', type=float, default=0.1)
    argparser.add_argument('--n_hid', type=int, default=256)
    argparser.add_argument('--n_layers', type=int, default=3)  # 300
    argparser.add_argument('--temp', type=float, default=1.0)
    argparser.add_argument('--uni_rand_pb', type=float, default=1.0) 
    
    # Environment 
    argparser.add_argument('--envsize', type=int, default=8)
    argparser.add_argument('--min_reward', type=float, default=1e-3)  # 1e-6
    argparser.add_argument('--enable_time', type=bool, default=False)
    argparser.add_argument('--consistent_signs', type=bool, default=True) 
    argparser.add_argument('--custom_reward_fn', type=callable, default=somitogenesis_reward_func)
    argparser.add_argument('--grid_bound', type=dict, default={
        'weight': {'min': -100, 'max': 100},     # For the 9 weight parameters
        'diagonal': {'min': -100, 'max': 100},    # For the 3 diagonal factors
    })
    argparser.add_argument('--actions_per_dim', type=dict, default={
        'weight': [1, 5, 25, -1, -5, -25],   # For the 9 weight parameters
        'diagonal': [1, 5, 25, -1, -5, -25],         # For the 3 diagonal factors
    })
    
    # argparser.add_argument('--n_nodes', type=int, default=3) # not used, can be infered from n_dims by solve quadratic
    # argparser.add_argument('--n_steps', type=int, default=2+6+10) 
    # argparser.add_argument('--n_dims', type=int, default=3**2+3)
    # argparser.add_argument('--steps_per_network', type=dict, default={1:2, 2:6, 3:10})
    
    # argparser.add_argument('--n_steps', type=int, default=8+24+40) 
    # argparser.add_argument('--n_dims', type=int, default=3**2+3)
    # argparser.add_argument('--steps_per_network', type=dict, default={1: 8, 2: 24, 3: 40})
        
    argparser.add_argument('--n_steps', type=int, default=(1+3+5+7+9+11+13)*4) 
    argparser.add_argument('--n_dims', type=int, default=7**2+7)
    argparser.add_argument('--steps_per_network', type=dict, default={1:1*4, 2:3*4, 3:5*4, 4:7*4, 5:9*4, 6:11*4, 7:13*4}) 

    args = argparser.parse_args()
    main(args)



