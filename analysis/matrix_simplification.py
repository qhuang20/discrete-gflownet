"""
Matrix Simplification using SVD for GFlowNet States

This module provides tools to simplify weight matrices in test_states using Singular Value 
Decomposition (SVD). The goal is to create sparser, more interpretable matrices while 
preserving the essential dynamics and reward properties.

Key Functions:
- simplify_test_state(): Apply SVD simplification to any test_state
- find_optimal_simplification(): Automatically find best rank/threshold parameters with optional visualization
- compare_test_states(): Compare performance of original vs simplified states
- batch_simplify(): Process multiple test_states

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
import time
from typing import List, Tuple, Dict, Optional, Any
import os
import sys

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from reward_func.evo_devo import weights_to_matrix, somitogenesis_reward_func, somitogenesis_sol_func
from graph.graph import draw_network_motif, plot_network_motifs_and_somites


def matrix_to_weights(W_matrix: np.ndarray) -> List[int]:
    """
    Inverse of weights_to_matrix function.
    """
    n_nodes = W_matrix.shape[0]
    weights = []
    
    if n_nodes == 1:
        return [round(W_matrix[0, 0])]
    
    if n_nodes == 2:
        # For 2x2: [w1,w2,w3,w4] -> [[w1,w4],[w3,w2]]
        weights = [
            round(W_matrix[0, 0]),  # w1
            round(W_matrix[1, 1]),  # w2  
            round(W_matrix[1, 0]),  # w3
            round(W_matrix[0, 1])   # w4
        ]
        return weights
    
    # For larger matrices, recursively extract the (n-1)x(n-1) inner matrix first
    prev_matrix = W_matrix[:n_nodes-1, :n_nodes-1]
    prev_weights = matrix_to_weights(prev_matrix)
    weights.extend(prev_weights)
    
    # Add diagonal element of nth node
    weights.append(round(W_matrix[n_nodes-1, n_nodes-1]))
    
    # Add rest of last row and column alternating
    for i in range(n_nodes-1):
        weights.append(round(W_matrix[n_nodes-1, i]))  # Last row
        weights.append(round(W_matrix[i, n_nodes-1]))  # Last column
    
    return weights


def apply_svd_simplification(W_original: np.ndarray, rank: int = 2, threshold: float = 1.0) -> np.ndarray:
    """
    Apply SVD simplification with given rank and threshold.
    """
    U, s, Vt = svd(W_original)
    print(s)
    
    # Low-rank approximation
    W_approx = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
    
    # Apply thresholding
    W_simplified = W_approx.copy()
    W_simplified[np.abs(W_simplified) < threshold] = 0  # Sets all "small" values to zero
    
    return W_simplified


def parse_test_state(test_state: List[float]) -> Tuple[int, List[float], List[float]]:
    n_nodes = int((-1 + (1 + 4*len(test_state))**0.5) / 2)
    n_weights = n_nodes * n_nodes
    weights = test_state[:n_weights]
    d_values = test_state[n_weights:]
    return n_nodes, weights, d_values











def simplify_test_state(test_state: List[float], rank: int = 2, threshold: float = 1.0, 
                       verbose: bool = True) -> Tuple[List[float], Dict[str, Any]]:
    """
    Apply SVD simplification to any test_state.
    
    Args:
        test_state: Complete test_state vector
        rank: Number of singular values to keep (lower = more simplified)
        threshold: Minimum absolute value to keep (higher = more sparse)
        verbose: Whether to print analysis
    
    Returns:
        Tuple of (simplified_test_state, metrics_dict)
    """
    # Parse input state
    n_nodes, weights_orig, d_values = parse_test_state(test_state)
    
    # Convert to matrix and simplify
    W_orig = weights_to_matrix(weights_orig).astype(float)
    W_simp = apply_svd_simplification(W_orig, rank=rank, threshold=threshold)
    
    # Convert back to weight vector
    weights_simp = matrix_to_weights(W_simp)
    test_state_simp = weights_simp + d_values
    
    # Calculate reward difference
    try:
        reward_orig = somitogenesis_reward_func(test_state, plot=False)
        reward_simp = somitogenesis_reward_func(test_state_simp, plot=False)
        reward_difference = abs(reward_orig - reward_simp)
    except Exception as e:
        # If reward function fails, set to None
        reward_orig = None
        reward_simp = None
        reward_difference = None
        if verbose:
            print(f"  Warning: Could not calculate reward difference: {e}")
    
    # Calculate metrics
    metrics = {
        'original_sparsity': np.sum(W_orig == 0) / W_orig.size,
        'simplified_sparsity': np.sum(W_simp == 0) / W_simp.size,
        'sparsity_improvement': (np.sum(W_simp == 0) - np.sum(W_orig == 0)) / W_orig.size,
        'reconstruction_error': np.linalg.norm(W_orig - W_simp, 'fro'),
        'reward_original': reward_orig,
        'reward_simplified': reward_simp,
        'reward_difference': reward_difference,
        'n_nodes': n_nodes,
        'rank_used': rank,
        'threshold_used': threshold,
        'original_matrix': W_orig,
        'simplified_matrix': W_simp
    }
    
    if verbose:
        print(f"SVD Simplification Results for {n_nodes}-node system:")
        print(f"  Rank: {rank}, Threshold: {threshold}")
        print(f"  Sparsity: {metrics['original_sparsity']:.1%} → {metrics['simplified_sparsity']:.1%}")
        print(f"  Improvement: +{metrics['sparsity_improvement']:.1%}")
        print(f"  Reconstruction error: {metrics['reconstruction_error']:.2f}")
        if metrics['reward_difference'] is not None:
            print(f"  Reward: {metrics['reward_original']:.3f} → {metrics['reward_simplified']:.3f}")
            print(f"  Reward difference: {metrics['reward_difference']:.3f}")
    
    return test_state_simp, metrics







def find_optimal_simplification(test_state: List[float], max_reward_diff: float = 9.0, 
                               verbose: bool = True, plot: bool = False,
                               save_path: Optional[str] = None) -> Tuple[Optional[List[float]], Optional[Dict], List[Dict]]:
    """
    Find the best rank and threshold combination for a given test_state.
    
    Args:
        test_state: Input state to simplify
        max_reward_diff: Maximum acceptable reward difference (prioritized over reconstruction error)
        verbose: Whether to print search results
        plot: Whether to create visualization plots of top simplification options
        save_path: Optional path to save plots
    
    Returns:
        Tuple of (best_simplified_state, best_params, all_results)
    """
    # Parse input for plotting
    n_nodes, weights, d_values = parse_test_state(test_state)
    W_original = weights_to_matrix(weights).astype(float)
    
    ranks = list(range(1, n_nodes + 1))  # Test all possible ranks from 1 to n_nodes
    thresholds = [0.0, 6.0, 12.0, 15.0, 21.0, 24.0, 27.0, 30.0, 33.0]  # Include 0.0 for baseline (no thresholding)
    
    results = []
    
    if verbose:
        print(f"Searching for optimal simplification (prioritizing reward preservation)...")
        print(f"{'Rank':<4} {'Thresh':<6} {'Sparsity':<8} {'RewDiff':<8} {'Error':<8} {'Score':<8}")
        print("-" * 55)
    
    for rank in ranks:
        for threshold in thresholds:
            try:
                simp_state, metrics = simplify_test_state(
                    test_state, rank=rank, threshold=threshold, verbose=False
                )
                
                # Calculate composite score (high sparsity, low reward difference prioritized)
                if metrics['reward_difference'] is not None:
                    # Primary constraint: reward difference
                    if metrics['reward_difference'] <= max_reward_diff:
                        score = metrics['simplified_sparsity'] - 0.1 * metrics['reward_difference']
                        # Minor penalty for reconstruction error (secondary consideration)
                        score -= 0.001 * metrics['reconstruction_error']
                    else:
                        score = -1  # Penalize high reward difference
                else:
                    # Fallback to reconstruction error if reward calculation failed
                    if metrics['reconstruction_error'] <= max_reward_diff * 10:  # Scale appropriately
                        score = metrics['simplified_sparsity'] - 0.01 * metrics['reconstruction_error']
                    else:
                        score = -1
                
                result = {
                    'rank': rank,
                    'threshold': threshold,
                    'simplified_state': simp_state,
                    'metrics': metrics,
                    'score': score
                }
                results.append(result)
                
                if verbose:
                    reward_diff_str = f"{metrics['reward_difference']:<8.1f}" if metrics['reward_difference'] is not None else f"{'N/A':<8}"
                    print(f"{rank:<4} {threshold:<6.1f} {metrics['simplified_sparsity']:<8.1%} "
                          f"{reward_diff_str} {metrics['reconstruction_error']:<8.1f} {score:<8.3f}")
                    
            except Exception as e:
                if verbose:
                    print(f"{rank:<4} {threshold:<6.1f} ERROR: {str(e)[:20]}")
    
    # Find best result - prioritize reward preservation
    valid_results = [r for r in results if r['score'] > -5] # assume all valid
    if not valid_results:
        if verbose:
            print(f"No valid simplifications found within reward difference limit of {max_reward_diff}!")
        return None, None, results
    
    best_result = max(valid_results, key=lambda x: x['score'])
    

    
    if plot and valid_results:
        # Filter for states to plot based on the specified criteria
        plottable_results = [
            r for r in valid_results 
            if r['metrics']['reward_difference'] is not None 
            and r['metrics']['simplified_sparsity'] >= 0.10 
            and r['metrics']['reward_difference'] <= max_reward_diff
        ]

        # Plot original + all filtered simplified states (original first)
        states_to_plot = [test_state] + [r['simplified_state'] for r in plottable_results]
        
        # Create titles for each plot
        additional_titles = ["Original State"]
        for r in plottable_results:
            sp = r['metrics']['simplified_sparsity']
            rd = r['metrics']['reward_difference']
            title = f"Rank {r['rank']}, Thr {r['threshold']:.1f}, Sparsity {sp:.1%}, RewDiff {rd:.2f}"
            additional_titles.append(title)
        
        # Determine save path for the grid plot
        grid_save_path = save_path
        if not grid_save_path:
            images_folder = "matrix_simplification_images"
            os.makedirs(images_folder, exist_ok=True)
            state_id = f"{len(test_state)}elem_{abs(hash(tuple(test_state))) % 100000:05d}"
            grid_save_path = os.path.join(images_folder, f"simplification_grid_{state_id}.png")

        filename = plot_network_motifs_and_somites(states_to_plot, grid_save_path, additional_titles=additional_titles)
        if verbose:
            print(f"Saved simplification plots to: {filename}")
    
    if verbose:
        print(f"\nOriginal test_state ({n_nodes}-node): {test_state}")
        
        # List all simplifications with >=10% sparsity and reward difference <= max_reward_diff
        print(f"\nAll simplifications with >=10% sparsity and reward difference <= {max_reward_diff}:")
        for r in results:
            sp = r['metrics']['simplified_sparsity']
            rd = r['metrics']['reward_difference']
            if rd is not None and sp >= 0.10 and rd <= max_reward_diff:
                print(f"  Rank {r['rank']}, Thresh {r['threshold']}: Sparsity {sp:.1%}, Reward diff {rd:.3f}")
                # print(f"    Simplified state: {r['simplified_state']}")
        
        print(f"\nBest option (prioritizing reward preservation):")
        print(f"  Rank {best_result['rank']}, Threshold {best_result['threshold']}")
        print(f"  Sparsity: {best_result['metrics']['simplified_sparsity']:.1%}")
        if best_result['metrics']['reward_difference'] is not None:
            print(f"  Reward difference: {best_result['metrics']['reward_difference']:.3f}")
        print(f"  Reconstruction error: {best_result['metrics']['reconstruction_error']:.2f}")
        print(f"  Score: {best_result['score']:.3f}")
    return best_result['simplified_state'], best_result, results






def compare_test_states(original: List[float], simplified: List[float], 
                       detailed: bool = False) -> Dict[str, float]:
    """
    Compare performance of original vs simplified test_states.
    
    Args:
        original: Original test_state
        simplified: Simplified test_state  
        detailed: If True, do detailed trajectory analysis
        
    Returns:
        Dictionary with comparison metrics
    """
    print("Comparing test_states performance...")
    
    # Quick reward comparison
    start_time = time.perf_counter()
    reward_orig = somitogenesis_reward_func(original, plot=False)
    time_orig = time.perf_counter() - start_time
    
    start_time = time.perf_counter()
    reward_simp = somitogenesis_reward_func(simplified, plot=False)
    time_simp = time.perf_counter() - start_time
    
    reward_diff = abs(reward_orig - reward_simp)
    reward_rel_diff = reward_diff / max(abs(reward_orig), 1e-6)
    
    print(f"Rewards: {reward_orig:.3f} → {reward_simp:.3f}")
    print(f"Difference: {reward_diff:.3f} ({reward_rel_diff:.1%})")
    print(f"Computation time: {time_orig:.4f}s → {time_simp:.4f}s")
    
    results = {
        'reward_original': reward_orig,
        'reward_simplified': reward_simp,
        'reward_difference': reward_diff,
        'reward_relative_difference': reward_rel_diff,
        'time_original': time_orig,
        'time_simplified': time_simp
    }
    
    if detailed:
        # Detailed trajectory comparison
        test_positions = [25, 50, 75]
        trajectory_diffs = []
        
        for pos in test_positions:
            t_orig, traj_orig, _ = somitogenesis_sol_func(original, cell_position=pos)
            t_simp, traj_simp, _ = somitogenesis_sol_func(simplified, cell_position=pos)
            
            if traj_orig.shape == traj_simp.shape:
                diff = np.linalg.norm(traj_orig - traj_simp)
                max_val = max(np.max(np.abs(traj_orig)), np.max(np.abs(traj_simp)))
                rel_diff = diff / max_val if max_val > 0 else 0
                trajectory_diffs.append(rel_diff)
        
        avg_traj_diff = np.mean(trajectory_diffs)
        print(f"Average trajectory difference: {avg_traj_diff:.1%}")
        results['average_trajectory_difference'] = avg_traj_diff
    
    return results






def batch_simplify(test_states: List[List[float]], rank: int = 2, threshold: float = 1.0,
                  verbose: bool = True) -> List[Tuple[List[float], Dict]]:
    """
    Apply simplification to multiple test_states.
    
    Args:
        test_states: List of test_states to simplify
        rank: SVD rank to use for all
        threshold: Threshold to use for all
        verbose: Whether to print progress
        
    Returns:
        List of (simplified_state, metrics) tuples
    """
    results = []
    
    if verbose:
        print(f"Batch simplifying {len(test_states)} test_states...")
        print(f"Using rank={rank}, threshold={threshold}")
    
    for i, test_state in enumerate(test_states):
        try:
            simplified, metrics = simplify_test_state(test_state, rank, threshold, verbose=False)
            results.append((simplified, metrics))
            
            if verbose:
                sparsity_improvement = metrics['sparsity_improvement']
                error = metrics['reconstruction_error']
                print(f"  {i+1:3d}: {metrics['n_nodes']}-node, +{sparsity_improvement:.1%} sparsity, error {error:.1f}")
                
        except Exception as e:
            if verbose:
                print(f"  {i+1:3d}: ERROR - {str(e)}")
            results.append((None, None))
    
    successful = len([r for r in results if r[0] is not None])
    if verbose:
        print(f"Successfully simplified {successful}/{len(test_states)} test_states")
    
    return results





if __name__ == "__main__":
    # Example usage
    print("Matrix Simplification Module")
    print("=" * 40)
    
    # Example test_state (7-node system)
    example_state = [37, -89, 88, 89, 76, 51, -56, 43, -57, 35, 1, -16, 36, 7, -53, 6, -36, 26, 31, 56, 32, -55, -51, 10, 31, 6, 56, 1, -50, -2, 1, -32, 1, -30, -5, 5, 80, -100, 58, 75, -50, -50, 0, -30, -75, 76, 100, -26, -5, -39, -18, -36, -25, 51, -61, 1]
    
    print(f"\nExample: Analyzing {len(example_state)}-element test_state")
    
    # Find optimal simplification with comprehensive visualization
    optimal_state, optimal_params, _ = find_optimal_simplification(
        example_state, max_reward_diff=9.0, verbose=True, plot=True
    )
    
    if optimal_state:
        print(f"\nPerformance comparison:")
        compare_test_states(example_state, optimal_state, detailed=False) 