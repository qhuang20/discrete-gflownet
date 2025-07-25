"""
Reward function configurations.
"""

# Import available reward functions
try:
    from reward_func.evo_devo import coord_reward_func, oscillator_reward_func, somitogenesis_reward_func, somitogenesis_sparsity_reward_func
except ImportError:
    # Fallback if reward_func is not available
    coord_reward_func = None
    oscillator_reward_func = None
    somitogenesis_reward_func = None
    somitogenesis_sparsity_reward_func = None

REWARD_FUNCTIONS = {
    'coord': coord_reward_func,
    'oscillator': oscillator_reward_func,
    'somitogenesis': somitogenesis_reward_func,
    'somitogenesis_sparsity': somitogenesis_sparsity_reward_func,
}

def get_reward_function(reward_name):
    """Get reward function by name."""
    if reward_name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {reward_name}. Available: {list(REWARD_FUNCTIONS.keys())}")
    
    reward_func = REWARD_FUNCTIONS[reward_name]
    if reward_func is None:
        raise ImportError(f"Reward function {reward_name} is not available. Check imports.")
    
    return reward_func 