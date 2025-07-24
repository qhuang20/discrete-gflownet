"""
Configuration for GridEnvLocal (local exploration from a given state).
"""

from .baseenv_config import BaseEnvConfig

class GridEnvLocalConfig(BaseEnvConfig):
    
    seed = 42
    env_type = 'GridEnvLocal' 
    
    # Grid bounds - using larger bounds for local exploration
    grid_bound = {
        'weight': {'min': -500, 'max': 500},
        'diagonal': {'min': -500, 'max': 500},
    }
    
    # Actions available based on initial state sign
    actions_per_dim = {
        'weight': {
            'positive': [50, -5],    # [slot_0, slot_1] for positive initial values
            'negative': [-50, 5]     # [slot_0, slot_1] for negative initial values
        },
        'diagonal': {
            'positive': [50, -5],    # [slot_0, slot_1] for positive initial values
            'negative': [-50, 5]     # [slot_0, slot_1] for negative initial values
        }
    } 
    


    n_workers = 8
    n_train_steps = 200
    log_freq = 100    
        
    # Initial state to start exploration from, example state:
    # initial_state = [37, -89, 88, 89, 76, 51, -56, 43, -57, 35, 1, -16, 36, 7, -53, 6, 0, 0, 31, 0, 32, 0, -51, 0, 31, 6, 56, 1, -50, -2, 1, -32, 1, -30, -5, 0, 80, -100, 58, 75, -50, -50, 0, -30, -75, 76, 100, -26, -5, -39, -18, -36, -25, 51, -61, 1]
    # initial_state=[160, -50, -110, 105, 5, 0, 50, 50, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -105, 150, 0, 0, 0, 0, 0]
    initial_state=[165, -120, -75, 175, 155, -185, 200, -165, 120, -110, 20, -105, -15, -55, 200, 160, 5, -15, -10, 160, 105, 55, 100, -150, 155, -150, -155, 55, 55, 5, -5, 10, -100, 0, 10, 50, -50, 50, 5, -5, -5, 50, 10, 50, 50, 0, 0, -50, 5, -200, 175, 125, -130, -50, 50, -5]
    
    # Environment parameters
    n_dims = 56        # 7x7 + 7 for 7-node network
    n_steps = 50      # Number of steps for local exploration
    
