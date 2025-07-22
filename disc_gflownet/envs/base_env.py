import numpy as np


class BaseEnv:
    def __init__(self, args):
        self.min_reward = args.min_reward
        self.custom_reward_func = args.custom_reward_fn
        
        self.n_dims = args.n_dims
        self.n_steps = args.n_steps
        self.enable_time = args.enable_time
        
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
      
    def _grid_state_to_encoding(self, state) -> np.ndarray:
        """
        Common grid state encoding method for environments that use weight/diagonal structure.
        
        Subclasses should set:
        - self.n_weight_params: number of weight parameters  
        - self.n_nodes: number of nodes
        - self.grid_bound: bounds dictionary with 'weight' and 'diagonal' keys
        """
        if self.enable_time:
            time_step, spatial_state = state
        else:
            spatial_state = state

        one_hots = []
        
        if self.enable_time:
            time_one_hot = np.zeros(self.n_steps + 1, dtype=int)
            time_one_hot[time_step] = 1
            one_hots.append(time_one_hot)
            
        # Encode weight dimensions
        weight_size = self.grid_bound['weight']['max'] - self.grid_bound['weight']['min'] + 1
        for dim_val in spatial_state[:self.n_weight_params]:
            one_hot = np.zeros(weight_size, dtype=int)
            one_hot[dim_val - self.grid_bound['weight']['min']] = 1
            one_hots.append(one_hot)
            
        # Encode diagonal dimensions
        diag_size = self.grid_bound['diagonal']['max'] - self.grid_bound['diagonal']['min'] + 1
        for dim_val in spatial_state[self.n_weight_params:]:
            one_hot = np.zeros(diag_size, dtype=int)
            one_hot[dim_val - self.grid_bound['diagonal']['min']] = 1
            one_hots.append(one_hot)
            
        return np.concatenate(one_hots)

    def _grid_encoding_to_state(self, encoding) -> np.ndarray:
        """
        Common grid encoding to state method for environments that use weight/diagonal structure.
        
        Subclasses should set:
        - self.n_weight_params: number of weight parameters
        - self.n_nodes: number of nodes  
        - self.grid_bound: bounds dictionary with 'weight' and 'diagonal' keys
        """
        spatial_state = np.zeros(self.n_dims, dtype=np.int32)
        offset = 0
        time_step = 0
        
        if self.enable_time:
            time_one_hot = encoding[:self.n_steps + 1]
            time_step = np.where(time_one_hot == 1)[0][0]
            offset = self.n_steps + 1
        
        # Extract weight dimensions
        weight_size = self.grid_bound['weight']['max'] - self.grid_bound['weight']['min'] + 1
        for dim in range(self.n_weight_params):
            start_idx = offset + dim * weight_size
            end_idx = start_idx + weight_size
            one_hot_segment = encoding[start_idx:end_idx]
            hot_idx = np.where(one_hot_segment == 1)[0][0]
            spatial_state[dim] = hot_idx + self.grid_bound['weight']['min']
        
        # Extract diagonal dimensions
        diag_size = self.grid_bound['diagonal']['max'] - self.grid_bound['diagonal']['min'] + 1
        offset = offset + self.n_weight_params * weight_size
        for dim in range(self.n_nodes):
            start_idx = offset + dim * diag_size
            end_idx = start_idx + diag_size
            one_hot_segment = encoding[start_idx:end_idx]
            hot_idx = np.where(one_hot_segment == 1)[0][0]
            spatial_state[dim + self.n_weight_params] = hot_idx + self.grid_bound['diagonal']['min']
                
        return (time_step, spatial_state) if self.enable_time else spatial_state


    
    
    
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
        spatial_state = np.int32([0] * self.n_dims) # coord origin
        self._step = 0
        self._state = (0, spatial_state) if self.enable_time else spatial_state
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

