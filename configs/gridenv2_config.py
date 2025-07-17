"""
Configuration for GridEnv2 (divide and conquer approach).
"""

from .baseenv_config import BaseEnvConfig

class GridEnv2Config(BaseEnvConfig):

    seed = 42
    env_type = 'GridEnv2' 
    actions_per_dim = {'weight': [1, 5, 25, -1, -5, -25], 'diagonal': [1, 5, 25, -1, -5, -25]}
    # actions_per_dim = {'weight': [5, 25, -5, -25], 'diagonal': [5, 25, -5, -25]}
    
    n_workers = 12
    n_train_steps = 200
    log_freq = 100 # <= n_train_steps
    
        
    
    # eg. GridEnv2 parameters 
    # n_dims = 3**2 + 3               # Dimension size
    # steps_per_network = {1: 8, 2: 24, 3: 40}  # Steps per network size
    # n_steps = 8 + 24 + 40           # Total steps across all networks
    
    n_dims = 56
    steps_per_network = {1: 4, 2: 12, 3: 20, 4: 28, 5: 36, 6: 44, 7: 52}
    n_steps = 196  # Sum of steps_per_network values
    
    # NOTE: GridEnv2 uses progressive exploration (divide & conquer), so the n_steps
    # constraint from GridEnv doesn't apply. Instead, ensure steps_per_network values
    # allow sufficient exploration within each network size before moving to the next.
    
    
    
