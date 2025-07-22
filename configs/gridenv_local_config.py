"""
Configuration for GridEnvLocal (local exploration from a given state).
"""

from .baseenv_config import BaseEnvConfig

class GridEnvLocalConfig(BaseEnvConfig):
    
    seed = 42
    env_type = 'GridEnvLocal' 
    
    n_workers = 12
    n_train_steps = 200
    log_freq = 100
    
    # Environment parameters
    n_dims = 56        # 7x7 + 7 for 7-node network
    n_steps = 100      # Number of steps for local exploration
    
    # Initial state to start exploration from
    # This is the example state from the notebook
    initial_state = [37, -89, 88, 89, 76, 51, -56, 43, -57, 35, 1, -16, 36, 7, -53, 6, 
                     0, 0, 31, 0, 32, 0, -51, 0, 31, 6, 56, 1, -50, -2, 1, -32, 
                     1, -30, -5, 0, 80, -100, 58, 75, -50, -50, 0, -30, -75, 76, 
                     100, -26, -5, -39, -18, -36, -25, 51, -61, 1]
    
    # Grid bounds - using larger bounds for local exploration
    grid_bound = {
        'weight': {'min': -1000, 'max': 1000},
        'diagonal': {'min': -1000, 'max': 1000},
    }
    
    # Actions available based on initial state sign
    actions_per_dim = {
        'weight': {
            'positive': [100, -10],    # [slot_0, slot_1] for positive initial values
            'negative': [-100, 10]     # [slot_0, slot_1] for negative initial values
        },
        'diagonal': {
            'positive': [100, -10],    # [slot_0, slot_1] for positive initial values
            'negative': [-100, 10]     # [slot_0, slot_1] for negative initial values
        }
    } 