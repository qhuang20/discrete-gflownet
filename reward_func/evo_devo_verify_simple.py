#!/usr/bin/env python3
"""
Simple test script for 2-node somitogenesis system.
Tests the exact mathematical equations derived from test_state = [100, 0, 40, -100, 30, 20]
"""

import numpy as np
import matplotlib.pyplot as plt
from evo_devo import somitogenesis_reward_func, weights_to_matrix

def test_2node_equations():
    """
    Test and verify the 2-node somitogenesis equations.
    """
    test_state = [100, 0, 40, -100, 30, 20]
    
    print("="*60)
    print("2-Node Somitogenesis System - Simple Test")
    print("="*60)
    print(f"Test state: {test_state}")
    
    # Parse the state
    n_nodes = int((-1 + (1 + 4*len(test_state))**0.5) / 2)  # = 2
    weights = test_state[:4]  # [100, 0, 40, -100]
    d_values = test_state[4:6]  # [30, 20]
    
    print(f"\nParsed parameters:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Weights: {weights}")
    print(f"  D values: {d_values}")
    
    # Transform to matrices
    W_original = weights_to_matrix(weights)  # [[100, -100], [40, 0]]
    W_scaled = W_original / 10               # [[10, -10], [4, 0]]
    D_scaled = np.diag(d_values) / 10        # [[3, 0], [0, 2]]
    D_ones = D_scaled @ np.ones(2)           # [3, 2]
    
    print(f"\nWeight matrix (original):\n{W_original}")
    print(f"Weight matrix (scaled):\n{W_scaled}")
    print(f"Diagonal matrix (scaled):\n{D_scaled}")
    print(f"D @ ones: {D_ones}")
    
    # Show the exact equations
    print(f"\n" + "="*60)
    print("EXACT ODE EQUATIONS")
    print("="*60)
    print("For cell i ∈ {0, 1, ..., 99} and time t:")
    print("")
    print("Gene 1:")
    print("dx_{i,1}/dt = σ(3·g_i(t) + 10·x_{i,1} - 10·x_{i,2}) - x_{i,1}")
    print("")
    print("Gene 2:")
    print("dx_{i,2}/dt = σ(2·g_i(t) + 4·x_{i,1} + 0·x_{i,2}) - x_{i,2}")
    print("")
    print("Where:")
    print("  g_i(t) = min(exp(0.02·i - 0.04·t), 1)")
    print("  σ(z) = 1/(1 + exp(5-z))")
    
    # Run simulation and get reward
    reward = somitogenesis_reward_func(test_state, plot=False)
    print(f"\nSimulation reward: {reward}")
    
    # Create a simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Gene 1 concentration
    reward1 = somitogenesis_reward_func(test_state, plot=True, ax=ax1)
    ax1.set_title(f'Gene 1 Concentration (Reward: {reward1})')
    
    # Show the parameter summary
    ax2.axis('off')
    summary_text = f"""
2-Node System Parameters

Test State: {test_state}

Weight Matrix (scaled):
{W_scaled}

Key Interactions:
• Gene 1 → Gene 1: +{W_scaled[0,0]:.1f} (self-activation)
• Gene 2 → Gene 1: {W_scaled[0,1]:.1f} (inhibition)  
• Gene 1 → Gene 2: +{W_scaled[1,0]:.1f} (activation)
• Gene 2 → Gene 2: {W_scaled[1,1]:.1f} (no interaction)

Diagonal Factors:
• Gene 1: {D_ones[0]:.1f}
• Gene 2: {D_ones[1]:.1f}

This creates a simple toggle switch:
- Gene 1 activates itself and Gene 2
- Gene 2 inhibits Gene 1
- Moving gradient creates wave patterns
"""
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('2node_simple_test.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return reward

def manual_equation_check():
    """
    Manually verify the equations at a specific point.
    """
    print(f"\n" + "="*60)
    print("MANUAL EQUATION VERIFICATION")
    print("="*60)
    
    # Pick a specific cell and time
    cell_i = 50  # Middle cell
    time_t = 10.0
    x1, x2 = 0.5, 0.3  # Example concentrations
    
    # Calculate gradient
    A, B = 0.02, 0.04
    g_i_t = min(np.exp(A * cell_i - B * time_t), 1)
    
    # Calculate z values for each gene
    z1 = 3 * g_i_t + 10 * x1 - 10 * x2  # Gene 1
    z2 = 2 * g_i_t + 4 * x1 + 0 * x2    # Gene 2
    
    # Calculate sigmoid values
    sigma1 = 1 / (1 + np.exp(5 - z1))
    sigma2 = 1 / (1 + np.exp(5 - z2))
    
    # Calculate derivatives
    dx1_dt = sigma1 - x1
    dx2_dt = sigma2 - x2
    
    print(f"Example calculation for cell {cell_i} at time {time_t}:")
    print(f"Current concentrations: x1={x1}, x2={x2}")
    print(f"Gradient: g_{cell_i}({time_t}) = {g_i_t:.4f}")
    print(f"")
    print(f"Gene 1:")
    print(f"  z1 = 3×{g_i_t:.4f} + 10×{x1} - 10×{x2} = {z1:.4f}")
    print(f"  σ(z1) = {sigma1:.4f}")
    print(f"  dx1/dt = {sigma1:.4f} - {x1} = {dx1_dt:.4f}")
    print(f"")
    print(f"Gene 2:")
    print(f"  z2 = 2×{g_i_t:.4f} + 4×{x1} + 0×{x2} = {z2:.4f}")
    print(f"  σ(z2) = {sigma2:.4f}")
    print(f"  dx2/dt = {sigma2:.4f} - {x2} = {dx2_dt:.4f}")

if __name__ == "__main__":
    # Run the test
    reward = test_2node_equations()
    
    # Manual verification
    manual_equation_check()
    
    print(f"\n" + "="*60)
    print(f"Test completed! Final reward: {reward}")
    print("Simple visualization saved as '2node_simple_test.png'")
    print("="*60) 
    
    
    