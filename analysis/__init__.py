"""
Analysis package for GFlowNet mode analysis and visualization.

This package contains tools for analyzing and visualizing discovered modes
from GFlowNet training, including diverse mode selection functionality and
matrix simplification using SVD.
"""

from .diversity_selection import (
    select_diverse_modes,
    select_by_structure_diversity,
    select_by_parameter_diversity,
    select_by_reward_diversity,
    select_by_topology_diversity,
    select_by_combined_diversity,
    analyze_diversity_metrics,
    print_diversity_comparison
)

from .matrix_simplification import (
    simplify_test_state,
    find_optimal_simplification,
    compare_test_states,
    batch_simplify,
    matrix_to_weights,
    apply_svd_simplification
)

__all__ = [
    # Diversity selection functions
    'select_diverse_modes',
    'select_by_structure_diversity',
    'select_by_parameter_diversity',
    'select_by_reward_diversity',
    'select_by_topology_diversity',
    'select_by_combined_diversity',
    'analyze_diversity_metrics',
    'print_diversity_comparison',
    
    # Matrix simplification functions
    'simplify_test_state',
    'find_optimal_simplification',
    'compare_test_states',
    'batch_simplify',
    'matrix_to_weights',
    'apply_svd_simplification'
] 