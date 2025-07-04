#!/usr/bin/env python3
"""
Test script to verify the environment configuration system works correctly.
"""

import sys
from configs import get_env, ENVS
from configs.reward_configs import get_reward_function, REWARD_FUNCTIONS

def test_envs():
    """Test all available environment configurations."""
    print("Testing Environment Configuration System")
    print("=" * 50)
    
    print(f"Available environments: {list(ENVS.keys())}")
    print(f"Available reward functions: {list(REWARD_FUNCTIONS.keys())}")
    print()
    
    # Test each environment configuration
    for env_name in ENVS.keys():
        print(f"Testing environment: {env_name}")
        try:
            env_class = get_env(env_name)
            env_config = env_class()
            
            # Print key attributes
            print(f"  - Environment type: {getattr(env_config, 'env_type', 'Not specified')}")
            print(f"  - n_dims: {env_config.n_dims}")
            print(f"  - n_steps: {env_config.n_steps}")
            print(f"  - method: {env_config.method}")
            print(f"  - n_train_steps: {env_config.n_train_steps}")
            
            # Check for environment-specific attributes
            if hasattr(env_config, 'steps_per_network'):
                print(f"  - steps_per_network: {env_config.steps_per_network}")
            if hasattr(env_config, 'max_nodes'):
                print(f"  - max_nodes: {env_config.max_nodes}")
                print(f"  - max_edges: {env_config.max_edges}")
            
            print("  ✓ Environment config loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading environment config: {e}")
            return False
        print()
    
    # Test reward functions
    print("Testing reward functions:")
    for reward_name in REWARD_FUNCTIONS.keys():
        try:
            reward_func = get_reward_function(reward_name)
            if reward_func is not None:
                print(f"  ✓ {reward_name}: {reward_func.__name__}")
            else:
                print(f"  ! {reward_name}: Not available (import issue)")
        except Exception as e:
            print(f"  ✗ {reward_name}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("Environment configuration system test completed!")
    return True

if __name__ == '__main__':
    success = test_envs()
    sys.exit(0 if success else 1) 