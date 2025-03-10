from .base_agent import BaseAgent
import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from ..nets.mlp import make_mlp
from ..utils.setting import get_device, tf



class DBFlowNetAgent(BaseAgent):
    """
    GFlowNet agent that uses the Detailed Balance (DB) objective function. 
    """
    def __init__(self, args, envs):
        super().__init__(args, envs)
        in_dim = self.tensor_dim
        out_dim = self.action_dim + self.action_dim + 1  # forward logits + backward logits + F(s) 
        self.model = make_mlp([in_dim] + [args.n_hid] * args.n_layers + [out_dim]).to(self.dev)
        print(self.model)
        print(self.env.print_actions() if hasattr(self.env, 'print_actions') else '') 
    
    def parameters(self):
        return self.model.parameters()



    def compute_batch_loss(self, batch, use_fldb=False):
        """
        Compute batch loss using either standard GFlowNet loss or FL-DB variant.
        Args:
            batch: List containing [batch_ss, batch_as, episode_lens, batch_rs]
            use_fldb: Boolean flag to use FL-DB variant loss computation
        Returns:
            List containing [avg_batch_loss, avg_batch_Z]
        """
        inf = 1e8
        batch_ss, batch_as, episode_lens, batch_rs = batch

        batch_loss = []
        batch_log_z = []
        for episode_idx in range(len(batch_ss)):
            episode_len = episode_lens[episode_idx]
            episode_states = batch_ss[episode_idx][:episode_len, :] 
            episode_actions = batch_as[episode_idx][:episode_len - 1, :] 
            episode_trajectory_rewards = batch_rs[episode_idx].squeeze(-1).to(self.dev) # move the reward to CUDA 
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
            batch_log_z.append(log_flow[0])
            
            # DB loss
            episode_loss = torch.zeros(episode_states.shape[0] - 1).to(self.dev)
            if use_fldb:
                episode_loss += fwd_logits
                episode_loss += log_flow[:-1]
                episode_loss -= bwd_logits
                episode_loss -= log_flow[1:]
                episode_loss -= episode_trajectory_rewards.log()
            else:
                log_flow = log_flow[:-1]
                episode_loss += fwd_logits
                episode_loss += log_flow
                episode_loss -= bwd_logits
                episode_loss[:-1] -= log_flow[1:]
                episode_loss[-1] -= episode_terminal_reward.log()
            batch_loss.append(episode_loss ** 2)

        avg_batch_loss = torch.cat(batch_loss).mean()
        avg_batch_log_z = torch.tensor(batch_log_z).mean()
        return [avg_batch_loss, torch.exp(avg_batch_log_z)]



