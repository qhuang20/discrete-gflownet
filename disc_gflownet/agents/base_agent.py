import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from ..nets.mlp import make_mlp
from ..utils.setting import get_device, tf
from itertools import count
from itertools import chain




class BaseAgent:
    """
    GFlowNet agent with no objective function implemented.  
    """
    def __init__(self, args, envs):
        self.dev = get_device()  
        self.envs = envs
        self.env = envs[0]
        self.action_mask_fn = self.env.get_masks
        self.tensor_dim = self.env.encoding_dim
        self.action_dim = self.env.action_dim
        
        in_dim = self.tensor_dim
        out_dim = self.action_dim + self.action_dim + (1) + 1  # forward logits + backward logits + (stop) + F(s) 
        self.model = None
        # print(self.model)
        # print(self.env.print_actions() if hasattr(self.env, 'print_actions') else '')

        self.explore_ratio = args.explore_ratio
        self.temp = args.temp
        self.uniform_pb = args.uni_rand_pb

        # Track unique state distribution and trajectory rewards
        self.encoding_to_state = self.env.encoding_to_state
        self.ep_last_state_counts = {}  # Counts occurrences of each episode last state.
        self.ep_last_state_trajectories = {}  # Store trajectories (states, actions, rewards) for each last state

    def parameters(self):
        return self.model.parameters()



    def sample_batch_episodes(self, mbsize): 
        inf = 1e8
        batch_ss = [[] for i in range(mbsize)]
        batch_as = [[] for i in range(mbsize)]
        batch_rs = [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        env_idx_terminal_reward_map = {} 
        notdone_env_idxs = [i for i in range(mbsize)]

        s_s = tf([i.reset() for i in self.envs])
        done_s = [False] * mbsize
        while not all(done_s):
            with torch.no_grad():
                pred = self.model(s_s)
                
                forward_mask = tf(self.action_mask_fn(s_s.cpu().numpy(), "fwd"))
                logits = torch.where(forward_mask.bool(), pred[..., :self.action_dim], -inf).log_softmax(1) # no stop implemented
                action_probs = (1 - self.explore_ratio) * (logits / self.temp).softmax(1) + self.explore_ratio * (forward_mask) / (forward_mask + 1e-7).sum(1).unsqueeze(1)

                a_s = action_probs.multinomial(1)
                a_s = a_s.squeeze(-1)

            step = [notdone_env.step(a) 
                    for notdone_env, a in zip([e for d, e in zip(done_s, self.envs) if not d], a_s)]

            for i, (curr_s, curr_a, (next_s, curr_r, done)) in enumerate(zip(s_s, a_s, step)):
                env_idx = notdone_env_idxs[i] # not done environment index
                env_idx_done_map[env_idx] = done
                batch_ss[env_idx].append(curr_s)
                batch_as[env_idx].append(curr_a.unsqueeze(-1))
                batch_rs[env_idx].append(tf([curr_r])) 
                if done:
                    batch_ss[env_idx].append(tf(next_s))
                    env_idx_terminal_reward_map[env_idx] = curr_r

            notdone_env_idxs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    notdone_env_idxs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done_s[j]}
            done_s = [bool(d or step[m[i]][2]) for i, d in enumerate(done_s)] # update done flags 
            s_s = tf([i[0] for i in step if not i[2]]) # update states in not done environments

        batch_steps = [len(batch_ss[i]) for i in range(len(batch_ss))]

        """post-process"""
        for i in range(len(batch_ss)):
            batch_ss[i] = torch.stack(batch_ss[i])
            batch_as[i] = torch.stack(batch_as[i])
            batch_rs[i] = torch.stack(batch_rs[i])
            assert batch_ss[i].shape[0] - batch_as[i].shape[0] == 1
        # Track unique state distribution and trajectory rewards
        for i in range(len(batch_rs)):
            # Get trajectory data
            ep_states = batch_ss[i].cpu().data.numpy()  # All states in trajectory
            ep_actions = batch_as[i].cpu().data.numpy()  # All actions in trajectory
            ep_rewards = batch_rs[i].cpu().data.numpy()  # All rewards in trajectory
            
            # Get final state
            encoding = ep_states[-1]
            env_state = self.encoding_to_state(encoding) 
            if self.env.enable_time:
                env_state = tuple(env_state[1])  # Use spatial state only for tracking
            else:
                env_state = tuple(env_state)
                
            # Update counts
            if env_state in self.ep_last_state_counts:
                self.ep_last_state_counts[env_state] += 1
            else:
                self.ep_last_state_counts[env_state] = 1
                self.ep_last_state_trajectories[env_state] = []
                
            # Store full trajectory
            trajectory = {
                'states': ep_states,
                'actions': ep_actions,
                'rewards': ep_rewards
            }
            self.ep_last_state_trajectories[env_state].append(trajectory)
        
        return [batch_ss, batch_as, batch_steps, batch_rs]





    def compute_batch_loss(self, batch):
        """
        Compute batch loss. (need to be implemented by subclass)
        """
        inf = 1e8
        batch_ss, batch_as, episode_lens, batch_rs = batch

        batch_loss = []
        batch_Z = []
        for episode_idx in range(len(batch_ss)):
            episode_len = episode_lens[episode_idx]
            episode_states = batch_ss[episode_idx][:episode_len, :] 
            episode_actions = batch_as[episode_idx][:episode_len - 1, :] 
            episode_trajectory_rewards = batch_rs[episode_idx].squeeze(-1)
            episode_terminal_reward = episode_trajectory_rewards[-1] # terminal reward can also be set as the sum of trajectory rewards 
            pred = self.model(episode_states)

            # P_F
            forward_mask = tf(self.action_mask_fn(episode_states.cpu().numpy(), "fwd"))
            fwd_logits = torch.where(forward_mask.bool(), pred[..., :self.action_dim], -inf).log_softmax(1)
            fwd_logits = fwd_logits[:-1, :].gather(1, episode_actions).squeeze(1) 

            # P_B
            backward_mask = tf(self.action_mask_fn(episode_states.cpu().numpy(), "bwd"))
            bwd_logits = torch.where(backward_mask.bool(), (0 if self.uniform_pb else 1) * pred[..., self.action_dim:2*self.action_dim], -inf).log_softmax(1)  
            bwd_logits = bwd_logits[1:, :].gather(1, episode_actions).squeeze(1) 

            # F(s) 
            log_flow = pred[..., -1]                
            batch_Z.append(torch.exp(log_flow[0]))

            # loss
            episode_loss = torch.zeros(episode_states.shape[0] - 1).to(self.dev)
            # ...
            batch_loss.append(episode_loss ** 2)

        avg_batch_loss = torch.cat(batch_loss).mean()
        avg_batch_Z = torch.tensor(batch_Z).mean()
        return [avg_batch_loss, avg_batch_Z]


