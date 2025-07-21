import numpy as np
from .base_env import BaseEnv
from ..utils.cache import LRUCache


class GridEnv(BaseEnv): 
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
        # Maximum number of nodes this subnetwork can have
        self.max_nodes = args.max_nodes
        # Maximum number of di-edges (on w) this subnetwork can have, < self.max_nodes * self.max_nodes
        self.max_edges = args.max_edges 
        
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
        
        # Validate n_steps constraint for effective exploration
        self._check_n_steps_constraint()
        
        # Calculate encoding dimension
        weight_encoding = (self.grid_bound['weight']['max'] - self.grid_bound['weight']['min'] + 1) * n_weight_params
        diag_encoding = (self.grid_bound['diagonal']['max'] - self.grid_bound['diagonal']['min'] + 1) * n_diag_params
        base_encoding_dim = weight_encoding + diag_encoding
        self.encoding_dim = base_encoding_dim + (self.n_steps + 1) if self.enable_time else base_encoding_dim
    
    def _check_n_steps_constraint(self):
        """Check if n_steps is within safe bounds for effective exploration."""
        # Calculate constraint: n_steps < (max_edges + max_nodes) * (grid_bound_max / max_action)
        grid_bound_max = self.grid_bound['weight']['max']  # Assuming symmetric bounds like -100 to 100
        max_action = max(abs(a) for a in self.actions_per_dim['weight'])
        total_params = self.max_edges + self.max_nodes
        max_safe_steps = int(total_params * (grid_bound_max / max_action))
        
        if self.n_steps >= max_safe_steps:
            print(f"\n⚠️  WARNING: n_steps ({self.n_steps}) may be too large for effective exploration!")
            print(f"   Recommended: n_steps < {max_safe_steps}")
            print(f"   Constraint: (max_edges + max_nodes) × (grid_bound_max ÷ max_action)")
            print(f"   Current: ({self.max_edges} + {self.max_nodes}) × ({grid_bound_max} ÷ {max_action}) = {max_safe_steps}")
            print(f"   Reason: Large n_steps → agent hits bounds frequently → limited exploration\n")
    
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
        
        # Determine which weights to allow based on max_nodes
        # For 1-node network, only allow action on w1
        # For 2-node network, allow actions on w1, w2, w3, w4
        # For k-node network (k > 2), allow actions on all weights up to current size
        allowed_weights = []
        
        if self.max_nodes == 1:
            allowed_weights = [0]  # Only w1
        elif self.max_nodes == 2:
            allowed_weights = [0, 1, 2, 3]  # w1, w2, w3, w4
        else:
            # For larger networks, allow all weights up to max_nodes^2
            allowed_weights = list(range(self.max_nodes * self.max_nodes))
        
        # Count current non-zero edges to enforce max_edges constraint
        non_zero_edges = sum(1 for i in range(n_weight_params) if spatial_state[i] != 0)
        
        # Handle weight actions
        weight_actions = len(self.actions_per_dim['weight'])
        for dim in allowed_weights:
            if dim >= n_weight_params:
                continue  # Skip if dimension is out of bounds
                
            for action_idx, action_val in enumerate(self.actions_per_dim['weight']):
                mask_idx = dim * weight_actions + action_idx
                next_state = spatial_state.copy()
                next_state[dim] += action_val if is_forward else -action_val
                
                # Check sign consistency when enabled
                if self.consistent_signs:
                    current_val = spatial_state[dim]
                    if current_val > 0 and action_val <= 0:
                        continue
                    elif current_val < 0 and action_val >= 0:
                        continue
                
                # Check max_edges constraint - if adding a new edge
                if spatial_state[dim] == 0 and next_state[dim] != 0:
                    if non_zero_edges >= self.max_edges:
                        continue  # Skip if we already have max edges
                
                if self._check_state_bounds(next_state[dim], is_weight=True):
                    mask[mask_idx] = True
        
        # Handle diagonal actions - only allow diagonals up to max_nodes
        diag_actions = len(self.actions_per_dim['diagonal'])
        base_idx = n_weight_params * weight_actions
        for dim in range(min(self.max_nodes, self.n_nodes)):
            for action_idx, action_val in enumerate(self.actions_per_dim['diagonal']):
                mask_idx = base_idx + dim * diag_actions + action_idx
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


