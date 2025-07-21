"""
Configuration for GridEnv (all at once approach).
"""

from .baseenv_config import BaseEnvConfig

class GridEnvConfig(BaseEnvConfig):
    
    seed = 42
    env_type = 'GridEnv' 
    
    n_workers = 12
    n_train_steps = 200
    log_freq = 100 # <= n_train_steps
    
    
    
    # eg. custom parameters for 7 node
    # n_dims = 7**2 + 7  # no need for --n_nodes, since it can be inferred from n_dims by solve quadratic equ
    # max_nodes = 3      # Maximum number of nodes this subnetwork can have
    # max_edges = 6      # Maximum number of di-edges (on w) this subnetwork can have, ≤ self.max_nodes * self.max_nodes
    # n_steps = (3+6)*(100//25) - 1  # < (max_edges + max_nodes) * (grid_bound / max(actions_per_dim))
    
    n_dims = 56        # Custom dimension size
    max_nodes = 2      # Maximum number of nodes this subnetwork can have
    max_edges = 4      # Maximum number of di-edges (on w) this subnetwork can have, ≤ self.max_nodes * self.max_nodes
    n_steps = 23       # Custom number of steps for each trajectory
    
    # IMPORTANT: To ensure effective exploration, keep n_steps under the constraint:
    # n_steps < (max_edges + max_nodes) * (grid_bound / max(actions_per_dim))
    # Current calculation: n_steps < (4+2) * (100/25) = 24 ✓ (n_steps=23 is safe)
    

    
    
    
    