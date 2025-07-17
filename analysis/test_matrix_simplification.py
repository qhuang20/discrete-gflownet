#!/usr/bin/env python3
"""
Example usage of Matrix Simplification module

This script demonstrates how to use the SVD-based matrix simplification tools
to analyze and simplify test_states discovered through GFlowNet training.

Run this script to see example outputs and learn how to use the functions.
"""

import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from analysis.matrix_simplification import (
    simplify_test_state,
    find_optimal_simplification,
    compare_test_states,
    batch_simplify
)


def example_basic_simplification():
    """Example 1: Basic simplification of a test_state"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Test State Simplification")
    print("=" * 60)
    
    # Example 7-node test_state (49 weights + 7 diagonal values = 56 elements)
    test_state = [37, -89, 88, 89, 76, 51, -56, 43, -57, 35, 1, -16, 36, 7, -53, 6, 
                  -36, 26, 31, 56, 32, -55, -51, 10, 31, 6, 56, 1, -50, -2, 1, -32, 
                  1, -30, -5, 5, 80, -100, 58, 75, -50, -50, 0, -30, -75, 76, 100, 
                  -26, -5, -39, -18, -36, -25, 51, -61, 1]
    
    print(f"Original test_state length: {len(test_state)}")
    
    # Apply basic simplification
    simplified_state, metrics = simplify_test_state(test_state, rank=2, threshold=1.0)
    
    print(f"Simplified test_state length: {len(simplified_state)}")
    print(f"Sparsity improvement: +{metrics['sparsity_improvement']:.1%}")
    print(f"Reconstruction error: {metrics['reconstruction_error']:.2f}")
    
    return test_state, simplified_state


def example_find_optimal():
    """Example 2: Find optimal simplification parameters"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Find Optimal Simplification Parameters")
    print("=" * 60)
    
    # Smaller example for faster computation (3-node system)
    test_state_3node = [10, -20, 30, -5, 15, -25, 5, 0, 10, 2, -3, 1]  # 3x3 + 3 diagonal
    
    print(f"Analyzing 3-node test_state: {test_state_3node}")
    
    # Find optimal parameters automatically
    optimal_state, optimal_params, all_results = find_optimal_simplification(
        test_state_3node, max_reward_diff=2.0, verbose=True
    )
    
    if optimal_state:
        print(f"\nOptimal simplified state: {optimal_state}")
        return test_state_3node, optimal_state
    else:
        print("No suitable simplification found!")
        return test_state_3node, None


def example_comprehensive_analysis():
    """Example 3: Comprehensive analysis with visualization"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Comprehensive Analysis with Visualization")
    print("=" * 60)
    
    # Medium example (4-node system)
    test_state_4node = [50, -30, 20, 10, 40, -60, 15, 25, -45, 35, 5, -15, 30, -20, 10, 0, 8, -5, 12, 3]
    
    print(f"Analyzing 4-node test_state with comprehensive analysis...")
    
    # Run comprehensive analysis with visualization
    optimal_state, optimal_params, all_results = find_optimal_simplification(
        test_state_4node, max_reward_diff=5.0, verbose=True, plot=True
    )
    
    if optimal_state:
        print(f"Analysis complete! Found {len(all_results)} simplification options.")
        print(f"Best option achieves {optimal_params['metrics']['simplified_sparsity']:.1%} sparsity")
        return optimal_params
    else:
        print("No suitable simplification found!")
        return None


def example_performance_comparison():
    """Example 4: Compare original vs simplified performance"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Performance Comparison")
    print("=" * 60)
    
    # Use the same state from example 1
    original_state = [37, -89, 88, 89, 76, 51, -56, 43, -57, 35, 1, -16, 36, 7, -53, 6, 
                      -36, 26, 31, 56, 32, -55, -51, 10, 31, 6, 56, 1, -50, -2, 1, -32, 
                      1, -30, -5, 5, 80, -100, 58, 75, -50, -50, 0, -30, -75, 76, 100, 
                      -26, -5, -39, -18, -36, -25, 51, -61, 1]
    
    # Get simplified version
    simplified_state, _ = simplify_test_state(original_state, rank=2, threshold=2.0, verbose=False)
    
    # Compare performance (this will test reward functions)
    try:
        comparison_results = compare_test_states(original_state, simplified_state, detailed=False)
        
        print(f"Performance preserved: reward difference = {comparison_results['reward_relative_difference']:.1%}")
        
        if comparison_results['reward_relative_difference'] < 0.1:
            print("✅ Excellent: Behavior well preserved!")
        elif comparison_results['reward_relative_difference'] < 0.2:
            print("✅ Good: Behavior mostly preserved")
        else:
            print("⚠️  Warning: Significant behavior change")
            
    except Exception as e:
        print(f"Performance comparison failed: {e}")
        print("(This is normal if reward functions require additional setup)")


def example_batch_processing():
    """Example 5: Batch process multiple test_states"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Batch Processing Multiple Test States")
    print("=" * 60)
    
    # Create multiple test_states of different sizes
    test_states = [
        # 2-node systems (4 weights + 2 diagonal = 6 elements)
        [10, -5, 8, -3, 2, 1],
        [15, -10, 12, -8, 3, -1],
        
        # 3-node systems (9 weights + 3 diagonal = 12 elements)  
        [20, -15, 10, 5, -8, 12, -6, 4, 9, 1, -2, 3],
        [25, -20, 15, -10, 8, -12, 6, -4, 11, 2, 1, -1],
    ]
    
    print(f"Batch processing {len(test_states)} test_states...")
    
    # Apply batch simplification
    results = batch_simplify(test_states, rank=2, threshold=1.5, verbose=True)
    
    # Analyze results
    successful = [r for r in results if r[0] is not None]
    print(f"\nSuccessfully processed {len(successful)}/{len(test_states)} test_states")
    
    if successful:
        avg_sparsity_improvement = sum(r[1]['sparsity_improvement'] for r in successful) / len(successful)
        avg_error = sum(r[1]['reconstruction_error'] for r in successful) / len(successful)
        print(f"Average sparsity improvement: +{avg_sparsity_improvement:.1%}")
        print(f"Average reconstruction error: {avg_error:.2f}")


def main():
    """Run all examples"""
    print("SVD Matrix Simplification Examples")
    print("=" * 60)
    print("This script demonstrates the matrix simplification functionality.")
    print("Each example shows different aspects of the SVD-based approach.")
    
    try:
        # Run examples
        original, simplified = example_basic_simplification()
        
        example_find_optimal()
        
        # Run visualization example
        example_comprehensive_analysis()
        
        example_performance_comparison()
        
        example_batch_processing()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED!")
        print("=" * 60)
        print("Key takeaways:")
        print("1. SVD simplification can significantly increase matrix sparsity")
        print("2. Rank 2-3 often provides good balance of simplicity vs accuracy")
        print("3. Threshold 1.0-2.0 typically creates good sparsity")
        print("4. Performance is usually well preserved with proper parameters")
        print("5. Batch processing enables analysis of multiple discovered modes")
        
        print("\nNext steps:")
        print("- Try the functions on your own discovered test_states")
        print("- Experiment with different rank and threshold values")
        print("- Use find_optimal_simplification() for automatic parameter selection")
        print("- Apply to high-reward states found through GFlowNet training")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("This might be due to missing dependencies or reward function setup.")
        print("The matrix simplification functions should still work for basic analysis.")


if __name__ == "__main__":
    main() 