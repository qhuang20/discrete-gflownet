#!/usr/bin/env python3
"""
Main training script for GFlowNet models.
Cleaner alternative to the original train.py with massive argument parser.
"""
import os
import sys
from argparse import ArgumentParser

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from configs import get_env, ENVS, get_reward_function, REWARD_FUNCTIONS
from trainer import GFlowNetTrainer

def create_args_from_config(env_class, reward_func_name):
    """Convert environment config class to args object with reward function."""
    from types import SimpleNamespace
    
    # Get all attributes from config class
    config_dict = {
        attr: getattr(env_class, attr) 
        for attr in dir(env_class) 
        if not attr.startswith('_')
    }
    
    # Add reward function
    config_dict['custom_reward_fn'] = get_reward_function(reward_func_name)
    config_dict['reward_func_name'] = reward_func_name
    
    return SimpleNamespace(**config_dict)

def main():
    """Main entry point with clean CLI interface."""
    print(f"Available OS CPU threads: {os.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    
    parser = ArgumentParser(description='GFlowNet for Genetic Circuits Design')
    
    parser.add_argument('--env', type=str, default='gridenv', 
                       choices=list(ENVS.keys()),
                       help=f'Environment configuration. Available: {list(ENVS.keys())}')
    
    parser.add_argument('--reward', type=str, default='somitogenesis',
                       choices=list(REWARD_FUNCTIONS.keys()),
                       help=f'Reward function. Available: {list(REWARD_FUNCTIONS.keys())}')
    
    parser.add_argument('--config', type=str, 
                       help='Optional: specify a custom config file path')
    
    parser.add_argument('--device', type=str, 
                       help='Override device (cpu/cuda)')
    
    parser.add_argument('--steps', type=int,
                       help='Override number of training steps') 
    
    parser.add_argument('--workers', type=int,
                       help='Override number of workers')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # TODO: Add support for custom config files
        raise NotImplementedError("Custom config files not yet supported")
    else:
        env_class = get_env(args.env)
        config = create_args_from_config(env_class, args.reward)
    
    # Apply CLI overrides
    if args.device:
        config.device = args.device
    if args.steps:
        config.n_train_steps = args.steps  
    if args.workers:
        config.n_workers = args.workers
        
    # Print configuration summary
    print(f"\n🚀 Starting GFlowNet Training")
    print(f"   Environment: {args.env} ({config.env_type})")
    print(f"   Reward: {args.reward}")
    print(f"   Method: {config.method}")
    print(f"   Steps: {config.n_train_steps}")
    print(f"   Workers: {config.n_workers}")
    print(f"   Device: {config.device}")
    print("   (Edit config files in configs/ to change parameters)\n")
    
    # Run training
    trainer = GFlowNetTrainer(config)
    results = trainer.train()
    
    print(f"\n✅ Training completed!")
    print(f"   Final loss: {results['losses'][-1]:.4f}")
    print(f"   Results saved to: {trainer.run_dir}")
    
    return results

if __name__ == '__main__':
    import torch
    main()



