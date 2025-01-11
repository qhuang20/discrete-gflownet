import numpy as np


class BaseEnv:
    def __init__(self, args):
        self.min_reward = args.min_reward
        self.custom_reward_func = args.custom_reward_fn
        
        self.n_dims = args.n_dims
        self.n_steps = args.n_steps
        
        # self.encoding_dim
        # self.action_dim

    def print_actions(self):
        pass
    
    def _get_action_mask(self, state, is_forward=True):
        # mask = np.zeros(self.action_dim, dtype=bool)
        # ...
        # return mask
        pass
    
    def state_to_encoding(self, state):
        pass

    def encoding_to_state(self, encoding):
        pass            
    
    def step(self, action_idx):
        # self._step += 1
        # self._state[action_idx] = 1
        
        # reward = self.custom_reward_func(state, action_idx, ...) if self.custom_reward_func is not None else self.min_reward 
        # done = self._step == self.n_steps
        # return self.obs(), reward, done
        pass
    
    
  

    def obs(self, state=None):    
        s = self._state if state is None else state
        return self.state_to_encoding(s)

    def reset(self):
        self._state = np.int32([0] * self.n_dims) # coord origin
        self._step = 0
        return self.obs()
    
    def get_forward_mask(self, state):
        return self._get_action_mask(state, is_forward=True)
    
    def get_backward_mask(self, state):
        return self._get_action_mask(state, is_forward=False)
    
    def get_masks(self, encodings, direction):
        # Handle a list of encodings with forward or backward masks
        masks = []
        mask_func = self.get_forward_mask if direction == "fwd" else self.get_backward_mask
        for encoding in encodings:
            state = self.encoding_to_state(encoding)
            mask = mask_func(state)
            masks.append(mask)
        return masks
    
    
