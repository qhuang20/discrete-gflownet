import numpy as np
from .base_env import BaseEnv



class GridEnv(BaseEnv): 
    def __init__(self, args):
        # super().__init__(args) 
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
        self.encoding_dim = (2 * self.grid_bound + 1) * self.n_dims if self.has_mixed_actions else (self.grid_bound + 1) * self.n_dims
        
        
    
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
    
    def state_to_encoding(self, state) -> np.ndarray:
        # Convert state coordinates to one-hot encoding per dimension
        one_hots = []
        for dim_val in state:
            # Create a one-hot vector with appropriate size
            if self.has_mixed_actions:
                # For mixed actions, shift values to be 0-indexed
                one_hot = np.zeros(2 * self.grid_bound + 1, dtype=int)
                one_hot[dim_val + self.grid_bound] = 1
            elif self.only_negative_actions:
                # For negative actions, shift values to be 0-indexed
                one_hot = np.zeros(self.grid_bound + 1, dtype=int)
                one_hot[dim_val + self.grid_bound] = 1
            else:
                # For positive-only actions
                one_hot = np.zeros(self.grid_bound + 1, dtype=int)
                one_hot[dim_val] = 1
            one_hots.append(one_hot)
        return np.concatenate(one_hots)

    def encoding_to_state(self, encoding) -> np.ndarray:
        # Convert one-hot encoding back to state coordinates
        state = np.zeros(self.n_dims, dtype=np.int32)
        dim_size = 2 * self.grid_bound + 1 if self.has_mixed_actions else self.grid_bound + 1
        
        for dim in range(self.n_dims):
            # Extract one-hot segment for this dimension
            start_idx = dim * dim_size
            end_idx = start_idx + dim_size
            one_hot_segment = encoding[start_idx:end_idx]
            # Find the index of 1 in the one-hot segment
            hot_idx = np.where(one_hot_segment == 1)[0][0]
            
            if self.has_mixed_actions:
                # Convert back from shifted index for mixed actions
                state[dim] = hot_idx - self.grid_bound
            elif self.only_negative_actions:
                # Convert back from shifted index for negative actions
                state[dim] = hot_idx - self.grid_bound
            else:
                # No shift needed for positive-only actions
                state[dim] = hot_idx
        return state
    
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
        mask = np.zeros(self.action_dim, dtype=bool)
        for action_idx in range(self.action_dim):
            dim = action_idx // len(self.actions_per_dim)
            action_val = self.actions_per_dim[action_idx % len(self.actions_per_dim)]
            next_state = state.copy()
            # For forward mask add the action, for backward mask subtract it
            next_state[dim] += action_val if is_forward else -action_val
            if self._check_state_bounds(next_state[dim]):
                mask[action_idx] = True
        return mask


  


    def step(self, a):
        # Convert flat action index to dimension and action value
        dim = a // len(self.actions_per_dim)  # Which dimension to modify
        action_idx = a % len(self.actions_per_dim)  # Which action to apply
        action_value = self.actions_per_dim[action_idx]  # The actual value to add
        
        # Update the state in the chosen dimension
        self._state[dim] += action_value
        self._step += 1

        # Episode ends if we've used all steps OR if there are no valid actions left
        forward_mask = self.get_forward_mask(self._state)
        done = (self._step == self.n_steps) or (not np.any(forward_mask))
        
        reward = self.custom_reward_func(self._state) + self.min_reward
        return self.obs(), reward, done



