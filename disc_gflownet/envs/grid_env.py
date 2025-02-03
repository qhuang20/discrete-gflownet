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
        self.actions_per_dim = args.actions_per_dim # Can be mixed positive/negative values
        self.action_dim = self.n_dims * len(self.actions_per_dim) # Total number of possible actions
        
        # Determine the action type 
        self.only_negative_actions = all(a < 0 for a in self.actions_per_dim)
        self.has_mixed_actions = any(a > 0 for a in self.actions_per_dim) and any(a < 0 for a in self.actions_per_dim)
        
        # Grid bounds
        self.grid_bound = args.grid_bound
        self.enable_time = args.enable_time
        self.consistent_signs = args.consistent_signs # New flag to enforce consistent signs per dimension
        
        # Calculate bits needed for each dimension
        if self.has_mixed_actions:
            self.bits_per_dim = int(np.ceil(np.log2(2 * self.grid_bound + 1)))
        else:
            self.bits_per_dim = int(np.ceil(np.log2(self.grid_bound + 1)))
            
        base_encoding_dim = self.bits_per_dim * self.n_dims
        self.encoding_dim = base_encoding_dim + (self.n_steps + 1) if self.enable_time else base_encoding_dim
    
    def print_actions(self):
        print("-"*42)
        action_names = []
        for dim in range(self.n_dims):
            for action in self.actions_per_dim:
                action_names.append(f"dim{dim} {action}")
        print("All available actions (action_names):", len(action_names))
        assert len(action_names) == self.action_dim
        print([f"({i}): {name}" for i, name in enumerate(action_names)])
        print("-"*42)   
    
    def _int_to_gray(self, n):
        """Convert an integer to Gray code"""
        return n ^ (n >> 1)
    
    def _gray_to_int(self, n):
        """Convert a Gray code back to integer"""
        mask = n
        while mask:
            mask >>= 1
            n ^= mask
        return n
    
    def state_to_encoding(self, state) -> np.ndarray:
        encoding = []
        
        # Add time encoding if enabled
        if self.enable_time:
            time_step, spatial_state = state
            time_one_hot = np.zeros(self.n_steps + 1, dtype=int)
            time_one_hot[time_step] = 1
            encoding.extend(time_one_hot)
        else:
            spatial_state = state
            
        # Convert each dimension to binary Gray code
        for dim_val in spatial_state:
            # Shift value to be non-negative
            if self.has_mixed_actions or self.only_negative_actions:
                dim_val += self.grid_bound
                
            # Convert to Gray code
            gray_val = self._int_to_gray(dim_val)
            
            # Convert to binary array
            binary = [(gray_val >> i) & 1 for i in range(self.bits_per_dim)]
            encoding.extend(binary)
            
        return np.array(encoding, dtype=int)

    def encoding_to_state(self, encoding) -> np.ndarray:
        offset = 0
        time_step = 0
        
        # Extract time if enabled
        if self.enable_time:
            time_one_hot = encoding[:self.n_steps + 1]
            time_step = np.where(time_one_hot == 1)[0][0]
            offset = self.n_steps + 1
        
        spatial_state = np.zeros(self.n_dims, dtype=np.int32)
        
        # Process each dimension
        for dim in range(self.n_dims):
            # Extract bits for this dimension
            start_idx = offset + dim * self.bits_per_dim
            end_idx = start_idx + self.bits_per_dim
            binary = encoding[start_idx:end_idx]
            
            # Convert binary array to integer
            gray_val = sum(int(bit) * (1 << i) for i, bit in enumerate(binary))
            
            # Convert Gray code to regular binary
            val = self._gray_to_int(gray_val)
            
            # Shift back if necessary
            if self.has_mixed_actions or self.only_negative_actions:
                val -= self.grid_bound
                
            spatial_state[dim] = val
                
        return (time_step, spatial_state) if self.enable_time else spatial_state
    
    def _check_state_bounds(self, state_val):
        """Helper method to check if a state value is within valid bounds"""
        if self.only_negative_actions:
            # For negative-only actions, valid range is [-grid_bound, 0]
            return -self.grid_bound <= state_val <= 0
        elif self.has_mixed_actions:
            # For mixed actions, valid range is [-grid_bound, grid_bound]
            return -self.grid_bound <= state_val <= self.grid_bound
        else:
            # For positive-only actions, valid range is [0, grid_bound]
            return 0 <= state_val <= self.grid_bound

    def _get_action_mask(self, state, is_forward=True):
        # Extract spatial state if time is enabled
        spatial_state = state[1] if self.enable_time else state
        
        mask = np.zeros(self.action_dim, dtype=bool)
        for action_idx in range(self.action_dim):
            dim = action_idx // len(self.actions_per_dim)
            action_val = self.actions_per_dim[action_idx % len(self.actions_per_dim)]
            next_state = spatial_state.copy()
            # For forward mask add the action, for backward mask subtract it
            next_state[dim] += action_val if is_forward else -action_val
            
            # Check sign consistency when enabled
            if self.consistent_signs and self.has_mixed_actions:
                current_val = spatial_state[dim]
                if current_val > 0:  # If positive, only allow positive actions
                    if action_val <= 0:
                        mask[action_idx] = False
                        continue
                elif current_val < 0:  # If negative, only allow negative actions
                    if action_val >= 0:
                        mask[action_idx] = False
                        continue
                # At zero, allow any action
            
            if self._check_state_bounds(next_state[dim]):
                mask[action_idx] = True
        return mask

    def step(self, a):
        # Convert flat action index to dimension and action value
        dim = a // len(self.actions_per_dim)  # Which dimension to modify
        action_idx = a % len(self.actions_per_dim)  # Which action to apply
        action_value = self.actions_per_dim[action_idx]  # The actual value to add
        
        # Update the state in the chosen dimension
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




