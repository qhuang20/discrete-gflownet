#!/usr/bin/env python3
"""
Test script to verify the config system works correctly.
"""

import sys
from configs import get_config, CONFIGS
from configs.reward_configs import get_reward_function, REWARD_FUNCTIONS

def test_configs():
    """Test all available configurations."""
    print("Testing Config System")
    print("=" * 50)
    
    print(f"Available configs: {list(CONFIGS.keys())}")
    print(f"Available reward functions: {list(REWARD_FUNCTIONS.keys())}")
    print()
    
    # Test each configuration
    for config_name in CONFIGS.keys():
        print(f"Testing config: {config_name}")
        try:
            config_class = get_config(config_name)
            config = config_class()
            
            # Print key attributes
            print(f"  - Environment type: {getattr(config, 'env_type', 'Not specified')}")
            print(f"  - n_dims: {config.n_dims}")
            print(f"  - n_steps: {config.n_steps}")
            print(f"  - method: {config.method}")
            print(f"  - n_train_steps: {config.n_train_steps}")
            
            # Check for environment-specific attributes
            if hasattr(config, 'steps_per_network'):
                print(f"  - steps_per_network: {config.steps_per_network}")
            if hasattr(config, 'max_nodes'):
                print(f"  - max_nodes: {config.max_nodes}")
                print(f"  - max_edges: {config.max_edges}")
            
            print("  ✓ Config loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading config: {e}")
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
    print("Config system test completed!")
    return True

if __name__ == '__main__':
    success = test_configs()
    sys.exit(0 if success else 1) 