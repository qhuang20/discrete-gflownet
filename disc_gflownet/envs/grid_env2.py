import numpy as np
from .base_env import BaseEnv
from ..utils.cache import LRUCache


class GridEnv2(BaseEnv): 
    def __init__(self, args):
        # super().__init__(args) 
        self.n_workers = args.n_workers 
        self.reward_cache = LRUCache(max_size=args.cache_max_size) 
        self.min_reward = args.min_reward
        self.custom_reward_func = args.custom_reward_fn
        
        self.n_steps = args.n_steps
        self.n_dims = args.n_dims
        
        # Infer n_nodes from n_dims by solving quadratic: n^2 + n - n_dims = 0
        self.n_nodes = int((-1 + (1 + 4*self.n_dims)**0.5) / 2)
        
        self.actions_per_dim = args.actions_per_dim # Dict with actions for weights and diagonals
        
        # Calculate total number of possible actions
        n_weight_params = self.n_nodes * self.n_nodes  # n^2 weight parameters
        n_diag_params = self.n_nodes  # n diagonal parameters
        self.action_dim = (len(self.actions_per_dim['weight']) * n_weight_params + 
                          len(self.actions_per_dim['diagonal']) * n_diag_params)
        
        # Grid bounds for weights and diagonal parameters
        self.grid_bound = args.grid_bound
        self.enable_time = args.enable_time
        self.consistent_signs = args.consistent_signs
        
        # Calculate encoding dimension
        weight_encoding = (self.grid_bound['weight']['max'] - self.grid_bound['weight']['min'] + 1) * n_weight_params
        diag_encoding = (self.grid_bound['diagonal']['max'] - self.grid_bound['diagonal']['min'] + 1) * n_diag_params
        base_encoding_dim = weight_encoding + diag_encoding
        self.encoding_dim = base_encoding_dim + (self.n_steps + 1) if self.enable_time else base_encoding_dim
        
        # Steps per network size for progressive search
        self.steps_per_network = args.steps_per_network
    
    def reset(self):
        spatial_state = np.int32([0] * self.n_dims) # coord origin
        self._step = 0
        # Steps per network size for progressive search
        self._step_in_current_network = 0
        self.current_network_size = 1  # Reset to 1-node network
        self._state = (0, spatial_state) if self.enable_time else spatial_state
        return self.obs()
    
        
    
    def print_actions(self):
        print("-"*42)
        action_names = []
        n_weight_params = self.n_nodes * self.n_nodes
        # Add weight actions
        for dim in range(n_weight_params):
            for action in self.actions_per_dim['weight']:
                action_names.append(f"weight{dim} {action}")
        # Add diagonal actions    
        for dim in range(self.n_nodes):
            for action in self.actions_per_dim['diagonal']:
                action_names.append(f"diag{dim} {action}")
        print("All available actions (action_names):", len(action_names))
        assert len(action_names) == self.action_dim
        print([f"({i}): {name}" for i, name in enumerate(action_names)])
        print("-"*42)   
    
    def state_to_encoding(self, state) -> np.ndarray:
        # Unpack time and spatial state if time is enabled
        if self.enable_time:
            time_step, spatial_state = state
        else:
            spatial_state = state

        # Convert state coordinates to one-hot encoding per dimension
        one_hots = []
        
        # Add time encoding if enabled
        if self.enable_time:
            time_one_hot = np.zeros(self.n_steps + 1, dtype=int)
            time_one_hot[time_step] = 1
            one_hots.append(time_one_hot)
            
        n_weight_params = self.n_nodes * self.n_nodes
        
        # Encode weight dimensions
        weight_size = self.grid_bound['weight']['max'] - self.grid_bound['weight']['min'] + 1
        for dim_val in spatial_state[:n_weight_params]:
            one_hot = np.zeros(weight_size, dtype=int)
            one_hot[dim_val - self.grid_bound['weight']['min']] = 1
            one_hots.append(one_hot)
            
        # Encode diagonal dimensions
        diag_size = self.grid_bound['diagonal']['max'] - self.grid_bound['diagonal']['min'] + 1
        for dim_val in spatial_state[n_weight_params:]:
            one_hot = np.zeros(diag_size, dtype=int)
            one_hot[dim_val - self.grid_bound['diagonal']['min']] = 1
            one_hots.append(one_hot)
            
        return np.concatenate(one_hots)

    def encoding_to_state(self, encoding) -> np.ndarray:
        spatial_state = np.zeros(self.n_dims, dtype=np.int32)
        offset = 0
        time_step = 0
        
        # Extract time if enabled
        if self.enable_time:
            time_one_hot = encoding[:self.n_steps + 1]
            time_step = np.where(time_one_hot == 1)[0][0]
            offset = self.n_steps + 1
        
        n_weight_params = self.n_nodes * self.n_nodes
        
        # Extract weight dimensions
        weight_size = self.grid_bound['weight']['max'] - self.grid_bound['weight']['min'] + 1
        for dim in range(n_weight_params):
            start_idx = offset + dim * weight_size
            end_idx = start_idx + weight_size
            one_hot_segment = encoding[start_idx:end_idx]
            hot_idx = np.where(one_hot_segment == 1)[0][0]
            spatial_state[dim] = hot_idx + self.grid_bound['weight']['min']
        
        # Extract diagonal dimensions
        diag_size = self.grid_bound['diagonal']['max'] - self.grid_bound['diagonal']['min'] + 1
        offset = offset + n_weight_params * weight_size
        for dim in range(self.n_nodes):
            start_idx = offset + dim * diag_size
            end_idx = start_idx + diag_size
            one_hot_segment = encoding[start_idx:end_idx]
            hot_idx = np.where(one_hot_segment == 1)[0][0]
            spatial_state[dim + n_weight_params] = hot_idx + self.grid_bound['diagonal']['min']
                
        return (time_step, spatial_state) if self.enable_time else spatial_state
    
    def _check_state_bounds(self, state_val, is_weight):
        """Helper method to check if a state value is within valid bounds"""
        bound = self.grid_bound['weight'] if is_weight else self.grid_bound['diagonal']
        return bound['min'] <= state_val <= bound['max']

    def _get_action_mask(self, state, is_forward=True):
        # Extract spatial state if time is enabled
        spatial_state = state[1] if self.enable_time else state
        mask = np.zeros(self.action_dim, dtype=bool)
        n_weight_params = self.n_nodes * self.n_nodes
                
        
        # Simple masking logic based on current network size
        # For a network of size k, we allow actions on specific weights
        allowed_weight_indices = []
        
        if self.current_network_size == 1:
            # For 1-node network, only allow action on w1
            allowed_weight_indices = [0]
        elif self.current_network_size == 2:
            # For 2-node network, allow actions on w2, w3, w4
            allowed_weight_indices = [1, 2, 3]
        else:
            # For k-node network (k > 2), allow actions on weights corresponding to the new node
            prev_weights_count = (self.current_network_size - 1) ** 2
            current_weights_count = self.current_network_size ** 2
            allowed_weight_indices = list(range(prev_weights_count, current_weights_count))
        
        # Handle weight actions 
        weight_actions = len(self.actions_per_dim['weight'])
        for dim_idx in allowed_weight_indices:
            for action_idx, action_val in enumerate(self.actions_per_dim['weight']):
                mask_idx = dim_idx * weight_actions + action_idx
                
                # Skip if mask_idx is out of bounds
                if mask_idx >= n_weight_params * weight_actions:
                    continue
                    
                next_state = spatial_state.copy()
                next_state[dim_idx] += action_val if is_forward else -action_val
                
                # Check sign consistency when enabled
                if self.consistent_signs:
                    current_val = spatial_state[dim_idx]
                    if current_val > 0 and action_val <= 0:
                        continue
                    elif current_val < 0 and action_val >= 0:
                        continue
                
                if self._check_state_bounds(next_state[dim_idx], is_weight=True):
                    mask[mask_idx] = True
        
        # Handle diagonal actions  
        diag_actions = len(self.actions_per_dim['diagonal'])
        base_idx = n_weight_params * weight_actions
        dim = self.current_network_size - 1  # 0-indexed, Only allow diagonal action for the current node
        for action_idx, action_val in enumerate(self.actions_per_dim['diagonal']):
            mask_idx = base_idx + dim * diag_actions + action_idx
            
            # Skip if mask_idx is out of bounds
            if mask_idx >= self.action_dim:
                continue
                
            next_state = spatial_state.copy()
            next_state[dim + n_weight_params] += action_val if is_forward else -action_val
            
            # Check sign consistency when enabled
            if self.consistent_signs:
                current_val = spatial_state[dim + n_weight_params]
                if current_val > 0 and action_val <= 0:
                    continue
                elif current_val < 0 and action_val >= 0:
                    continue
            
            if self._check_state_bounds(next_state[dim + n_weight_params], is_weight=False):
                mask[mask_idx] = True
                    
        return mask

    def step(self, a):
        # Determine if action is for weight or diagonal parameter
        weight_actions = len(self.actions_per_dim['weight'])
        n_weight_params = self.n_nodes * self.n_nodes
        total_weight_actions = n_weight_params * weight_actions
        
        if a < total_weight_actions:
            # Weight parameter action
            dim = a // weight_actions
            action_idx = a % weight_actions
            action_value = self.actions_per_dim['weight'][action_idx]
        else:
            # Diagonal parameter action
            diag_actions = len(self.actions_per_dim['diagonal'])
            a_adjusted = a - total_weight_actions
            dim = n_weight_params + (a_adjusted // diag_actions)
            action_idx = a_adjusted % diag_actions
            action_value = self.actions_per_dim['diagonal'][action_idx]
        
        # Update the state
        if self.enable_time:
            spatial_state = self._state[1].copy()
            spatial_state[dim] += action_value
            self._state = (self._step + 1, spatial_state)
        else:
            self._state[dim] += action_value
            
        self._step += 1
        self._step_in_current_network += 1
        
        # Check if we need to move to the next network size
        max_steps_for_current = self.steps_per_network.get(self.current_network_size, 0)
        if self._step_in_current_network >= max_steps_for_current and self.current_network_size < self.n_nodes:
            self.current_network_size += 1
            self._step_in_current_network = 0

        # Episode ends if we've used all steps OR if there are no valid actions left
        forward_mask = self.get_forward_mask(self._state)
        done = (self._step == self.n_steps) or (not np.any(forward_mask))
        
        state_for_reward = self._state[1] if self.enable_time else self._state

        if self.n_workers == 1:
            # Use cached reward if available, otherwise compute and cache it
            state_key = tuple(state_for_reward)
            if state_key in self.reward_cache:
                reward = self.reward_cache[state_key]
            else:
                reward = self.custom_reward_func(state_for_reward) + self.min_reward
                self.reward_cache[state_key] = reward
        else:
            # Return dummy reward - multiprocessing takes care of the rest
            reward = -1
            
        return self.obs(), reward, done



