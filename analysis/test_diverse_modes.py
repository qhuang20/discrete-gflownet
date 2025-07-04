#!/usr/bin/env python3
"""
Test script for diverse modes selection functionality.
This script demonstrates how to select diverse modes from a set of discovered modes.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.diversity_selection import select_diverse_modes, analyze_diversity_metrics, print_diversity_comparison
from graph.graph import plot_network_motifs_and_somites

def create_test_modes():
    """Create a set of test modes with different characteristics."""
    modes_dict = {}
    
    # Create some diverse test states
    test_states = [
        # 2-node systems
        [100, -50, 50, -25],  # Strong activation, weak inhibition
        [50, -100, 25, -50],  # Strong inhibition, weak activation
        [0, 0, 100, -100],    # No self-interactions, strong cross-interactions
        [100, 100, 0, 0],     # Strong self-activation, no cross-interactions
        [-50, -50, -50, -50], # All inhibitory
        [75, 25, 25, 75],     # Balanced positive interactions
        
        # 3-node systems (9 weights + 3 d_values = 12 parameters)
        [100, 0, 0, 0, 100, 0, 0, 0, 100, 50, 50, 50],  # Identity matrix with d-values
        [0, 100, 0, 0, 0, 100, 100, 0, 0, 25, 25, 25],  # Circular activation
        [100, -50, -50, -50, 100, -50, -50, -50, 100, 0, 0, 0],  # Self-activation with cross-inhibition
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100],  # No weights, only d-values
        [75, 25, 25, 25, 75, 25, 25, 25, 75, 10, 10, 10],  # Balanced positive
        [-25, -25, -25, -25, -25, -25, -25, -25, -25, -10, -10, -10],  # All negative
    ]
    
    # Create modes_dict with mock information
    for i, state in enumerate(test_states):
        mode_tuple = tuple(state)
        modes_dict[mode_tuple] = {
            'reward': 10.0 - i * 0.5,  # Decreasing rewards
            'step': i * 100,
            'states': [state],
            'rewards': [10.0 - i * 0.5]
        }
    
    return modes_dict

def test_diversity_selection():
    """Test different diversity selection methods."""
    print("Creating test modes...")
    modes_dict = create_test_modes()
    
    print(f"Created {len(modes_dict)} test modes")
    
    # Test different diversity metrics
    diversity_metrics = ['structure', 'parameters', 'rewards', 'topology', 'combined']
    
    for metric in diversity_metrics:
        print(f"\nTesting {metric} diversity selection...")
        diverse_states = select_diverse_modes(modes_dict, n_diverse=6, diversity_metric=metric)
        
        print(f"Selected {len(diverse_states)} diverse modes using {metric} metric:")
        for i, state in enumerate(diverse_states):
            state_tuple = tuple(state)
            if state_tuple in modes_dict:
                info = modes_dict[state_tuple]
                print(f"  Mode {i+1}: Reward={info['reward']:.2f}, State={state[:4]}...")
        
        # Plot the diverse modes
        plot_path = f"test_diverse_modes_{metric}.png"
        plot_network_motifs_and_somites(diverse_states, save_path=plot_path)
        print(f"  Plot saved to: {plot_path}")

def run_diversity_analysis():
    """Analyze how different diversity metrics perform."""
    modes_dict = create_test_modes()
    
    # Use the new module's analysis function
    results = analyze_diversity_metrics(modes_dict, n_diverse=6)
    
    # Print comparison using the new module's function
    print_diversity_comparison(results)
    
    return results

if __name__ == "__main__":
    print("Testing Diverse Modes Selection")
    print("=" * 50)
    
    # Test basic functionality
    test_diversity_selection()
    
    # Analyze different metrics
    print("\n" + "=" * 50)
    print("Analyzing Diversity Metrics")
    print("=" * 50)
    run_diversity_analysis()
    
    print("\nTest completed!") 