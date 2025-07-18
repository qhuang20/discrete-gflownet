"""
Base configuration for GFlowNet training.
Contains default parameters that are shared across different environments.
"""

class BaseEnvConfig:
    # Training
    device = 'cpu'  # cuda
    progress = True
    seed = 42
    n_train_steps = 1000
    n_workers = 8
    cache_max_size = 10_000  # cache will be used when n_workers == 1
    log_freq = 100
    log_flag = True
    mbsize = 8
    
    # Model
    method = 'fldb'  # 'tb', 'db', 'fldb'
    explore_ratio = 0.05
    learning_rate = 1e-3
    tb_lr = 0.01
    tb_z_lr = 0.1
    n_hid = 256
    n_layers = 3
    temp = 1.0
    uni_rand_pb = 1.0
    
    # Environment base settings
    envsize = 8
    min_reward = 1e-3
    enable_time = False
    consistent_signs = True
    
    # Grid bounds (shared across environments)
    # NOTE: grid_bound = max value (assuming symmetric bounds). Affects n_steps constraint in GridEnv.
    grid_bound = {
        'weight': {'min': -100, 'max': 100},     # For the 9 weight parameters (grid_bound = 100)
        'diagonal': {'min': -100, 'max': 100},    # For the 3 diagonal factors (grid_bound = 100)
    }
    
    # Actions per dimension (shared across environments)
    # NOTE: Larger action values allow more steps within the constraint
    actions_per_dim = {
        'weight': [5, 25, -5, -25],   # For the 9 weight parameters
        'diagonal': [5, 25, -5, -25], # For the 3 diagonal factors
    }
    
    # Default environment parameters (to be overridden by specific configs)
    n_dims = None
    n_steps = None
    env_type = None 