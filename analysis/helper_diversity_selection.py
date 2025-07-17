#!/usr/bin/env python3
"""
Diversity selection functions for GFlowNet mode analysis.
This module provides various methods to select diverse modes from discovered solutions.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from graph.graph import extract_network_parameters
from reward_func.evo_devo import weights_to_matrix


def select_diverse_modes(modes_dict, n_diverse=12, diversity_metric='combined'):
    """
    Select diverse modes based on different criteria.
    
    Args:
        modes_dict: Dictionary of modes with their information
        n_diverse: Number of diverse modes to select
        diversity_metric: Metric to use for diversity ('structure', 'parameters', 'rewards', 'topology', 'combined')
    
    Returns:
        list: List of diverse mode states
    """
    if len(modes_dict) <= n_diverse:
        return [list(mode) for mode in modes_dict.keys()]
    
    modes_list = list(modes_dict.keys())
    
    if diversity_metric == 'structure':
        # Select based on network structure diversity (weight matrix patterns)
        diverse_indices = select_by_structure_diversity(modes_list, n_diverse)
    elif diversity_metric == 'parameters':
        # Select based on parameter value diversity
        diverse_indices = select_by_parameter_diversity(modes_list, n_diverse)
    elif diversity_metric == 'rewards':
        # Select based on reward distribution diversity
        diverse_indices = select_by_reward_diversity(modes_dict, n_diverse)
    elif diversity_metric == 'topology':
        # Select based on network topology (connectivity patterns)
        diverse_indices = select_by_topology_diversity(modes_list, n_diverse)
    else:  # combined
        # Use a combination of metrics
        diverse_indices = select_by_combined_diversity(modes_dict, n_diverse)
    
    return [list(modes_list[i]) for i in diverse_indices]


def select_by_structure_diversity(modes_list, n_diverse):
    """Select diverse modes based on network structure (weight matrix patterns)."""
    
    # Convert states to weight matrices and flatten them
    weight_matrices = []
    for mode in modes_list:
        weights, _, _, n_nodes, _ = extract_network_parameters(list(mode))
        weight_matrix = weights_to_matrix(weights)
        weight_matrices.append(weight_matrix.flatten())
    
    weight_matrices = np.array(weight_matrices)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(weight_matrices)
    
    # Greedy selection: start with first mode, then add most dissimilar
    selected_indices = []
    remaining_indices = list(range(len(modes_list)))
    
    # Start with the first mode
    selected_indices.append(remaining_indices.pop(0))
    
    # Select remaining modes
    for _ in range(n_diverse - 1):
        if not remaining_indices:
            break
            
        # Find the mode most dissimilar to all selected modes
        max_min_similarity = -1
        best_idx = remaining_indices[0]
        
        for idx in remaining_indices:
            # Calculate minimum similarity to any selected mode
            min_similarity = min(similarity_matrix[idx, selected_indices])
            if min_similarity > max_min_similarity:
                max_min_similarity = min_similarity
                best_idx = idx
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return selected_indices


def select_by_parameter_diversity(modes_list, n_diverse):
    """Select diverse modes based on parameter value diversity."""
    
    # Convert states to numpy arrays
    states_array = np.array([list(mode) for mode in modes_list])
    
    # Use K-means clustering to find diverse clusters
    kmeans = KMeans(n_clusters=min(n_diverse, len(modes_list)), random_state=42)
    cluster_labels = kmeans.fit_predict(states_array)
    
    # Select one mode from each cluster (closest to cluster center)
    selected_indices = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            # Select the mode closest to cluster center
            cluster_center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(states_array[cluster_indices] - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected_indices.append(closest_idx)
    
    # If we have fewer clusters than requested, add more diverse modes
    while len(selected_indices) < n_diverse and len(selected_indices) < len(modes_list):
        remaining_indices = [i for i in range(len(modes_list)) if i not in selected_indices]
        if remaining_indices:
            selected_indices.append(remaining_indices[0])
    
    return selected_indices[:n_diverse]


def select_by_reward_diversity(modes_dict, n_diverse):
    """Select diverse modes based on reward distribution."""
    rewards = [modes_dict[mode]['reward'] for mode in modes_dict.keys()]
    modes_list = list(modes_dict.keys())
    
    # Create reward bins and select from different bins
    rewards_array = np.array(rewards)
    bins = np.linspace(rewards_array.min(), rewards_array.max(), n_diverse + 1)
    bin_indices = np.digitize(rewards_array, bins) - 1
    
    selected_indices = []
    for bin_id in range(n_diverse):
        bin_modes = np.where(bin_indices == bin_id)[0]
        if len(bin_modes) > 0:
            # Select the mode with highest reward in this bin
            bin_rewards = rewards_array[bin_modes]
            best_in_bin = bin_modes[np.argmax(bin_rewards)]
            selected_indices.append(best_in_bin)
    
    # If we have fewer bins than requested, add more modes
    while len(selected_indices) < n_diverse and len(selected_indices) < len(modes_list):
        remaining_indices = [i for i in range(len(modes_list)) if i not in selected_indices]
        if remaining_indices:
            selected_indices.append(remaining_indices[0])
    
    return selected_indices[:n_diverse]


def select_by_topology_diversity(modes_list, n_diverse):
    """Select diverse modes based on network topology (connectivity patterns)."""
    
    # Convert states to adjacency matrices (binary connectivity)
    adjacency_matrices = []
    for mode in modes_list:
        weights, _, _, n_nodes, _ = extract_network_parameters(list(mode))
        weight_matrix = weights_to_matrix(weights)
        # Create binary adjacency matrix (1 if connection exists, 0 otherwise)
        adjacency_matrix = (weight_matrix != 0).astype(float)
        adjacency_matrices.append(adjacency_matrix.flatten())
    
    adjacency_matrices = np.array(adjacency_matrices)
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(adjacency_matrices)
    
    # Greedy selection: start with first mode, then add most dissimilar
    selected_indices = []
    remaining_indices = list(range(len(modes_list)))
    
    # Start with the first mode
    selected_indices.append(remaining_indices.pop(0))
    
    # Select remaining modes
    for _ in range(n_diverse - 1):
        if not remaining_indices:
            break
            
        # Find the mode most dissimilar to all selected modes
        max_min_similarity = -1
        best_idx = remaining_indices[0]
        
        for idx in remaining_indices:
            # Calculate minimum similarity to any selected mode
            min_similarity = min(similarity_matrix[idx, selected_indices])
            if min_similarity > max_min_similarity:
                max_min_similarity = min_similarity
                best_idx = idx
        
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    return selected_indices


def select_by_combined_diversity(modes_dict, n_diverse):
    """Select diverse modes using a combination of structure and reward diversity."""
    modes_list = list(modes_dict.keys())
    
    # Get structure diversity scores
    structure_indices = select_by_structure_diversity(modes_list, n_diverse)
    
    # Get reward diversity scores
    reward_indices = select_by_reward_diversity(modes_dict, n_diverse)
    
    # Combine both sets and select unique indices
    combined_indices = list(set(structure_indices + reward_indices))
    
    # If we have more than needed, prioritize by reward
    if len(combined_indices) > n_diverse:
        rewards = [modes_dict[modes_list[i]]['reward'] for i in combined_indices]
        sorted_pairs = sorted(zip(combined_indices, rewards), key=lambda x: x[1], reverse=True)
        combined_indices = [idx for idx, _ in sorted_pairs[:n_diverse]]
    
    # If we have fewer than needed, add more from structure diversity
    while len(combined_indices) < n_diverse and len(combined_indices) < len(modes_list):
        remaining_indices = [i for i in range(len(modes_list)) if i not in combined_indices]
        if remaining_indices:
            combined_indices.append(remaining_indices[0])
    
    return combined_indices[:n_diverse]


def analyze_diversity_metrics(modes_dict, n_diverse=6):
    """
    Analyze how different diversity metrics perform.
    
    Args:
        modes_dict: Dictionary of modes with their information
        n_diverse: Number of diverse modes to select for analysis
    
    Returns:
        dict: Dictionary with analysis results for each metric
    """
    metrics = ['structure', 'parameters', 'rewards', 'topology', 'combined']
    results = {}
    
    for metric in metrics:
        diverse_states = select_diverse_modes(modes_dict, n_diverse=n_diverse, diversity_metric=metric)
        rewards = [modes_dict[tuple(state)]['reward'] for state in diverse_states]
        
        results[metric] = {
            'states': diverse_states,
            'rewards': rewards,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
    
    return results


def print_diversity_comparison(results):
    """
    Print a formatted comparison of different diversity metrics.
    
    Args:
        results: Dictionary returned by analyze_diversity_metrics
    """
    print("\nDiversity Metrics Comparison:")
    print("-" * 80)
    print(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8}")
    print("-" * 80)
    
    for metric, r in results.items():
        print(f"{metric:<12} {r['mean_reward']:<8.2f} {r['std_reward']:<8.2f} "
              f"{r['min_reward']:<8.2f} {r['max_reward']:<8.2f}")


if __name__ == "__main__":
    # Example usage and testing
    print("Diversity Selection Module")
    print("=" * 40)
    print("This module provides functions for selecting diverse modes from GFlowNet results.")
    print("Available diversity metrics:")
    print("- structure: Based on weight matrix patterns")
    print("- parameters: Based on parameter value clustering")
    print("- rewards: Based on reward distribution")
    print("- topology: Based on network connectivity patterns")
    print("- combined: Combination of structure and reward diversity") 