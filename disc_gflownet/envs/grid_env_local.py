import numpy as np
from .base_env import BaseEnv
from ..utils.cache import LRUCache


class GridEnvLocal(BaseEnv): 
    """
    Local exploration environment that starts from a given state.
    
    Action rules:
    - For zero initial values: no actions allowed further (stay zero)
    - For positive initial weights: first action can be +100 (slot 0) or -10 (slot 1)
    - For negative initial weights: first action can be -100 (slot 0) or +10 (slot 1)  
    - Action slot consistency: once slot_0 or slot_1 is chosen, only that slot is allowed for that dimension
    """
    
    def __init__(self, args):
        self.n_workers = args.n_workers 
        self.reward_cache = LRUCache(max_size=args.cache_max_size) 
        self.min_reward = args.min_reward
        self.custom_reward_func = args.custom_reward_fn
        
        self.n_steps = args.n_steps
        self.n_dims = args.n_dims
        
        # Infer n_nodes from n_dims by solving quadratic: n^2 + n - n_dims = 0
        self.n_nodes = int((-1 + (1 + 4*self.n_dims)**0.5) / 2)
        self.n_weight_params = self.n_nodes * self.n_nodes
        self.n_diag_params = self.n_nodes
        
        # Initial state to start exploration from
        self.initial_state = args.initial_state if hasattr(args, 'initial_state') else np.zeros(self.n_dims, dtype=np.int32)
        
        # Get actions from config - should have 'positive' and 'negative' keys for each type
        self.actions_per_dim = args.actions_per_dim
        
        # Track which dimensions have taken their first action and in what direction
        # None = no action yet, 'slot_0' = using first action slot, 'slot_1' = using second action slot
        self.action_directions = [None] * self.n_dims
        
        # Calculate total number of possible actions - FIXED BUG: Use consistent indexing with other envs
        # We use 2 slots per dimension (matching the hardcoded actions_per_dim = 2 in step/mask methods)
        self.slots_per_dim = 2
        self.action_dim = self.slots_per_dim * self.n_dims
        
        # Grid bounds for weights and diagonal parameters
        self.grid_bound = args.grid_bound
        self.enable_time = args.enable_time
        
        # Calculate encoding dimension
        weight_encoding = (self.grid_bound['weight']['max'] - self.grid_bound['weight']['min'] + 1) * self.n_weight_params
        diag_encoding = (self.grid_bound['diagonal']['max'] - self.grid_bound['diagonal']['min'] + 1) * self.n_diag_params
        base_encoding_dim = weight_encoding + diag_encoding
        self.encoding_dim = base_encoding_dim + (self.n_steps + 1) if self.enable_time else base_encoding_dim
    
    def reset(self):
        """Reset to the initial state instead of origin."""
        self._step = 0
        # Reset action directions tracking
        self.action_directions = [None] * self.n_dims
        # Start from the given initial state
        spatial_state = self.initial_state.copy()
        self._state = (0, spatial_state) if self.enable_time else spatial_state
        return self.obs()
    
    def set_initial_state(self, state):
        """Set a new initial state for exploration."""
        if len(state) != self.n_dims:
            raise ValueError(f"State must have {self.n_dims} dimensions, got {len(state)}")
        self.initial_state = np.array(state, dtype=np.int32)
        # Reset to use the new initial state
        return self.reset()
    
    def print_actions(self):
        print("-"*50)
        print("GridEnvLocal: Actions depend on initial state and first action taken")
        print("Weight actions:")
        print(f"  For positive initial values: {self.actions_per_dim['weight']['positive']}")
        print(f"  For negative initial values: {self.actions_per_dim['weight']['negative']}")
        print("Diagonal actions:")
        print(f"  For positive initial values: {self.actions_per_dim['diagonal']['positive']}")
        print(f"  For negative initial values: {self.actions_per_dim['diagonal']['negative']}")
        print(f"Total action dimension: {self.action_dim}")
        print(f"Action indexing: {self.slots_per_dim} slots per dimension")
        print("-"*50)   
    
    def state_to_encoding(self, state) -> np.ndarray:
        """Convert state to one-hot encoding using common grid method"""
        return self._grid_state_to_encoding(state)

    def encoding_to_state(self, encoding) -> np.ndarray:
        """Convert one-hot encoding back to state using common grid method"""
        return self._grid_encoding_to_state(encoding)
    
    def _check_state_bounds(self, state_val, is_weight, dim_idx=None):
        """Helper method to check if a state value is within valid bounds"""
        bound = self.grid_bound['weight'] if is_weight else self.grid_bound['diagonal']
        
        # Check global bounds
        if not (bound['min'] <= state_val <= bound['max']):
            return False
            
        # Check zero-crossing constraint based on initial value
        if dim_idx is not None:
            initial_val = self.initial_state[dim_idx]
            # Don't allow positive initial values to go negative
            if initial_val > 0 and state_val < 0:
                return False
            # Don't allow negative initial values to go positive  
            if initial_val < 0 and state_val > 0:
                return False
                
        return True

    def _get_available_actions(self, dim_idx):
        """Get available actions for a dimension based on initial value only."""
        # Get the initial value for this dimension
        initial_val = self.initial_state[dim_idx]
        
        # If initial value is zero, no actions are allowed (keep it zero)
        if initial_val == 0:
            return []
        
        # Determine action type (weight or diagonal)
        is_weight = dim_idx < self.n_weight_params
        action_type = 'weight' if is_weight else 'diagonal'
        
        # Get the appropriate action list based on initial value sign
        if initial_val > 0:
            return self.actions_per_dim[action_type]['positive']
        elif initial_val < 0:
            return self.actions_per_dim[action_type]['negative']
        else:
            return []  # This shouldn't happen due to check above, but for safety

    def _get_action_mask(self, state, is_forward=True):
        """Generate action mask based on current state and action history."""
        spatial_state = state[1] if self.enable_time else state
        mask = np.zeros(self.action_dim, dtype=bool)
        
        for dim_idx in range(self.n_dims):
            current_val = spatial_state[dim_idx]
            
            # Get all available actions for this dimension (based on initial value only)
            available_actions = self._get_available_actions(dim_idx)
            
            # Skip if no actions available (e.g., zero initial values)
            if not available_actions:
                continue
                
            # Determine which slots are available based on action direction history
            if self.action_directions[dim_idx] is None:
                # No direction set yet - both slots potentially available
                available_slots = list(range(min(len(available_actions), self.slots_per_dim)))
            elif self.action_directions[dim_idx] == 'slot_0':
                # Only slot 0 allowed
                available_slots = [0]
            else:  # 'slot_1'
                # Only slot 1 allowed
                available_slots = [1]
            
            # Check each available slot
            for slot in available_slots:
                if slot < len(available_actions):
                    action_val = available_actions[slot]
                    mask_idx = dim_idx * self.slots_per_dim + slot
                    
                    # Check bounds for the resulting state
                    is_weight = dim_idx < self.n_weight_params
                    next_val = current_val + (action_val if is_forward else -action_val)
                    
                    if self._check_state_bounds(next_val, is_weight, dim_idx):
                        mask[mask_idx] = True
                    
        return mask

    def step(self, a):
        """Execute action and update state."""
        # Decode action index - FIXED to be consistent with action_dim calculation
        dim_idx = a // self.slots_per_dim
        action_slot = a % self.slots_per_dim
        
        if dim_idx >= self.n_dims:
            raise ValueError(f"Invalid action {a}: dimension index {dim_idx} >= {self.n_dims}")
        
        # Get current value
        spatial_state = self._state[1].copy() if self.enable_time else self._state.copy()
        current_val = spatial_state[dim_idx]
        
        # Get all available actions for this dimension and extract the action value
        all_actions = self._get_available_actions(dim_idx)
        action_value = all_actions[action_slot]
        
        # Update action direction if this is the first action for this dimension
        if self.action_directions[dim_idx] is None:
            self.action_directions[dim_idx] = f'slot_{action_slot}'
        
        # Update the state
        spatial_state[dim_idx] += action_value
        
        if self.enable_time:
            self._state = (self._step + 1, spatial_state)
        else:
            self._state = spatial_state
            
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