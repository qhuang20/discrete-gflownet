import numpy as np
from .base_env import BaseEnv



class SetEnv(BaseEnv):
    def __init__(self, args):
        # self.min_reward = args.min_reward
        self.custom_reward_func = args.custom_reward_fn
        self.reward_set_size = args.reward_set_size
        
        self.n_dims = args.action_dim
        self.n_steps = args.set_size
        
        self.encoding_dim = self.n_dims
        self.action_dim = self.n_dims
        
    def print_actions(self):
        print("number of actions: ", self.n_dims)
        
    def state_to_encoding(self, state):
        return state

    def encoding_to_state(self, encoding):
        return encoding
        
    def get_forward_mask(self, state):
        return 1 - state

    def get_backward_mask(self, state):
        return state
    


    

    
    def step(self, action_idx):
        self._state[action_idx] = 1
        self._step += 1

        done = self._step == self.n_steps
        reward = self.custom_reward_func(action_idx, self.reward_set_size)
        return self.obs(), reward, done
    
    
