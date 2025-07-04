"""
Analysis package for GFlowNet mode analysis and visualization.

This package contains tools for analyzing and visualizing discovered modes
from GFlowNet training, including diverse mode selection functionality.
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

__all__ = [
    'select_diverse_modes',
    'select_by_structure_diversity',
    'select_by_parameter_diversity',
    'select_by_reward_diversity',
    'select_by_topology_diversity',
    'select_by_combined_diversity',
    'analyze_diversity_metrics',
    'print_diversity_comparison'
] 