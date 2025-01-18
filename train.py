import numpy as np
import argparse
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import argparse
import time
import datetime
import pickle

from disc_gflownet.utils.setting import set_seed, set_device, tf
from disc_gflownet.utils.plotting import plot_loss_curve
from disc_gflownet.utils.logging import log_arguments, log_training_loop
from disc_gflownet.utils.cache import LRUCache
from disc_gflownet.agents.tbflownet_agent import TBFlowNetAgent
from disc_gflownet.agents.dbflownet_agent import DBFlowNetAgent
from disc_gflownet.envs.grid_env import GridEnv
from disc_gflownet.envs.set_env import SetEnv

from reward_func.evo_devo import oscillator_reward_func, somitogenesis_reward_func

import os

from functools import partial


from multiprocessing import Pool

def compute_reward(curr_ns, env, reward_func):
    curr_ns_state = env.encoding_to_state(curr_ns)

    return reward_func(curr_ns_state) + env.min_reward



def main(args):
    global losses, zs, agent
    
    assert args.envsize == args.mbsize
    set_seed(args.seed)
    set_device(torch.device(args.device))
    
    # Environment setup 
    envs = [GridEnv(args) for _ in range(args.envsize)]
    
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
    zs = []  # only for tb
    for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):
        experiences = agent.sample_batch_episodes(args.mbsize)

        """ 
        # v1 
        for mb in range(args.mbsize):
            curr_s_all = experiences[0][mb].numpy()
            curr_ns_all = curr_s_all[1:]

            compute_reward_partial = partial(compute_reward, env=envs[mb], reward_func=args.custom_reward_fn)

            env_pool = Pool(processes=200)

            curr_r_env = list(env_pool.imap(compute_reward_partial, curr_ns_all, chunksize=20))
            curr_r_env = np.asarray(curr_r_env)
            curr_r_env = torch.from_numpy(curr_r_env)

            experiences[3][mb] = curr_r_env
        """

        curr_ns_all = np.zeros((args.mbsize, args.n_steps, envs[0].encoding_dim))
        for mb in range(args.mbsize):
            curr_ns_all[mb] = experiences[0][mb].numpy()[1:]
        curr_ns_all = curr_ns_all.reshape(args.mbsize*args.n_steps, envs[0].encoding_dim)

        compute_reward_partial = partial(compute_reward, env=envs[0], reward_func=args.custom_reward_fn)
        
        with Pool(processes=200) as env_pool:
            curr_r_env = list(env_pool.imap(compute_reward_partial, curr_ns_all, chunksize=1)) #tqdm(, total=args.mbsize*args.n_steps))
        curr_r_env = np.asarray(curr_r_env)
        curr_r_env = curr_r_env.reshape(args.mbsize, args.n_steps, 1)

        for mb in range(args.mbsize):
            experiences[3][mb] = torch.from_numpy(curr_r_env[mb])
        
        #exit()

        if args.method == 'fldb':
            loss, z = agent.compute_batch_loss(experiences, use_fldb=True) 
        else:
            loss, z = agent.compute_batch_loss(experiences) 
            
        losses.append(loss.item())
        zs.append(z.item()) 

        loss.backward()
        opt.step()
        opt.zero_grad() 
        
        if i % args.log_freq == 0 and args.log_flag:
            log_training_loop(log_filename, agent, i)

    return losses, zs, agent





    # # Save 
    # save_variables(
    #     run_dir,
    #     {"losses": losses, "zs": zs, "agent": agent}
    # )



if __name__ == "__main__":

    #parser = argparse.Namespace(description="GFlowNet for Genetic Circuits Design.")
    #args = parser.parse_args()

    args = argparse.Namespace(
    device='cpu',
    progress=True,
    seed=0,
    n_train_steps=2000,  # 2000
    log_freq=20,  # 1000
    log_flag=True,
    mbsize=16,
    # Model
    method='fldb', 
    learning_rate=1e-3,
    tb_lr=0.01,
    tb_z_lr=0.1,
    n_hid=256,
    n_layers=3,
    explore_ratio=0.35,  # 0.0625
    temp=1.,
    uni_rand_pb=1,
    # Env
    envsize=16,
    min_reward=0.0000001,
    cache_max_size=10_000,  # 10_000
    custom_reward_fn=somitogenesis_reward_func,
    n_steps=62,  # 8 * 9 = 72
    n_dims=9,  # 9,  
    actions_per_dim=[1, 5, 25, -1, -5, -25], # if inhomogenous, need to be a closed symmetrical group.
    consistent_signs=True,
    grid_bound=200,  # 100,
    enable_time=False  # True
    )

    main(args)
