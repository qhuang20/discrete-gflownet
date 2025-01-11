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



class TBFlowNetAgent(BaseAgent):
    """
    GFlowNet agent that uses the Trajectory Balance (TB) objective function. 
    """
    def __init__(self, args, envs):
        super().__init__(args, envs)
        in_dim = self.tensor_dim
        out_dim = self.action_dim + self.action_dim  # forward logits + backward logits
        self.model = make_mlp([in_dim] + [args.n_hid] * args.n_layers + [out_dim]).to(self.dev)
        self.log_z = nn.Parameter(torch.zeros((1,)).to(self.dev))   
        print(self.model)
        print(self.env.print_actions() if hasattr(self.env, 'print_actions') else '') 


    def parameters(self):
        return self.model.parameters()

    
    
    def compute_batch_loss(self, batch):
        inf = 1e8
        batch_ss, batch_as, episode_lens, batch_rs = batch

        batch_loss = []
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
            sum_fwd_logits = torch.sum(fwd_logits)

            # P_B
            backward_mask = tf(self.action_mask_fn(episode_states.cpu().numpy(), "bwd"))
            bwd_logits = torch.where(backward_mask.bool(), (0 if self.uniform_pb else 1) * pred[..., self.action_dim:2*self.action_dim], -inf).log_softmax(1)  
            bwd_logits = bwd_logits[1:, :].gather(1, episode_actions).squeeze(1) 
            sum_bwd_logits = torch.sum(bwd_logits)

            # TB loss
            episode_loss = (self.log_z + sum_fwd_logits - episode_terminal_reward.log() - sum_bwd_logits) ** 2
            batch_loss.append(episode_loss)
                    
        avg_batch_loss = torch.cat(batch_loss).mean()
        return [avg_batch_loss, torch.exp(self.log_z)] 


