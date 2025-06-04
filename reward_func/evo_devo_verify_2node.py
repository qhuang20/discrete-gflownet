import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from evo_devo import somitogenesis_reward_func, weights_to_matrix, sigmoid

def test_2node_somitogenesis():
    """
    Test and visualize the 2-node somitogenesis system with specific parameters.
    
    The test state [100, 0, 40, -100, 30, 20] represents:
    - Weights: [100, 0, 40, -100] -> Matrix [[100, -100], [40, 0]] -> Scaled [[10, -10], [4, 0]]
    - D values: [30, 20] -> Diagonal [[3, 0], [0, 2]]
    - S values: [1, 1] (fixed)
    
    The exact ODEs are:
    Gene 1: dx_{i,1}/dt = sigmoid(3*g_i(t) + 10*x_{i,1} - 10*x_{i,2}) - x_{i,1}
    Gene 2: dx_{i,2}/dt = sigmoid(2*g_i(t) + 4*x_{i,1}) - x_{i,2}
    
    Where g_i(t) = min(exp(0.02*i - 0.04*t), 1)
    """
    
    # Test state parameters
    test_state = [100, 0, 40, -100, 30, 20]
    
    print("="*60)
    print("2-Node Somitogenesis System Test")
    print("="*60)
    print(f"Test state: {test_state}")
    
    # Parse parameters
    n_nodes = int((-1 + (1 + 4*len(test_state))**0.5) / 2)
    n_weights = n_nodes * n_nodes
    weights = test_state[:n_weights]
    d_values = test_state[n_weights:n_weights+n_nodes]
    s_values = [1] * n_nodes
    
    print(f"Number of nodes: {n_nodes}")
    print(f"Weights: {weights}")
    print(f"D values: {d_values}")
    print(f"S values: {s_values}")
    
    # Show weight matrix transformation
    W_original = weights_to_matrix(weights)
    W_scaled = W_original / 10  # WEIGHT_SCALE = 10
    print(f"\nWeight matrix (original):\n{W_original}")
    print(f"Weight matrix (scaled by 1/10):\n{W_scaled}")
    
    # Show diagonal matrix
    D_scaled = np.diag(d_values) / 10  # DIAGONAL_SCALE = 10
    D_ONES = D_scaled @ np.ones(n_nodes)
    print(f"\nDiagonal matrix (scaled by 1/10):\n{D_scaled}")
    print(f"D @ ones = {D_ONES}")
    
    # Print the exact ODE equations
    print("\n" + "="*60)
    print("EXACT ODE EQUATIONS")
    print("="*60)
    print("For each cell i ∈ {0, 1, ..., 99} and time t:")
    print("\nGene 1:")
    print("dx_{i,1}/dt = 1/(1 + exp(5 - (3*g_i(t) + 10*x_{i,1} - 10*x_{i,2}))) - x_{i,1}")
    print("\nGene 2:")
    print("dx_{i,2}/dt = 1/(1 + exp(5 - (2*g_i(t) + 4*x_{i,1} + 0*x_{i,2}))) - x_{i,2}")
    print("\nWhere:")
    print("g_i(t) = min(exp(0.02*i - 0.04*t), 1)")
    print("Sigmoid: σ(z) = 1/(1 + exp(5-z))")
    
    # Run the simulation and measure performance
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    start_time = time.perf_counter()
    reward = somitogenesis_reward_func(test_state, plot=False)
    end_time = time.perf_counter()
    
    print(f"Reward: {reward}")
    print(f"Simulation time: {end_time - start_time:.4f} seconds")
    
    return test_state, reward

def visualize_2node_system(test_state):
    """
    Create comprehensive visualizations of the 2-node system.
    """
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Main heatmap for gene 1
    ax1 = plt.subplot(2, 3, 1)
    reward = somitogenesis_reward_func(test_state, plot=True, ax=ax1)
    ax1.set_title(f'Gene 1 Concentration\nReward: {reward}', fontsize=12)
    
    # 2. Simulate and plot gene 2 concentration
    ax2 = plt.subplot(2, 3, 2)
    
    # Extract parameters for manual simulation
    n_nodes = 2
    weights = test_state[:4]
    d_values = test_state[4:6]
    
    # System parameters
    N_CELLS = 100
    N_SIMTIME = 90
    N_TIMEPOINTS = 200
    WEIGHT_SCALE = 10
    DIAGONAL_SCALE = 10
    
    # Pre-compute parameters
    W = weights_to_matrix(weights) / WEIGHT_SCALE
    D = np.diag(d_values) / DIAGONAL_SCALE
    D_ONES = D @ np.ones(n_nodes)
    A, B = 0.1/5, 0.2/5
    S = np.ones(n_nodes)
    
    x0 = np.full(N_CELLS * n_nodes, 0.1)
    positions = np.arange(N_CELLS).reshape(-1, 1)
    
    def n_node_system(t, x, W):
        x_reshaped = x.reshape(-1, n_nodes)
        g = np.minimum(np.exp(A * positions - B * t), 1)
        z = g * D_ONES + x_reshaped @ W.T
        sigmoid_z = sigmoid(z)
        decay = x_reshaped * S
        return (sigmoid_z - decay).flatten()
    
    # Simulate
    t = np.linspace(0, N_SIMTIME, N_TIMEPOINTS)
    sol = solve_ivp(
        lambda t, x: n_node_system(t, x, W),
        (t[0], t[-1]),
        x0,
        t_eval=t,
        method='RK45',
        rtol=1e-3,
        atol=1e-6
    )
    
    # Reshape solution
    solution = sol.y.T.reshape(len(t), N_CELLS, n_nodes)
    x2_concentration = solution[:, :, 1]  # Gene 2
    
    # Plot gene 2 heatmap
    im2 = ax2.imshow(x2_concentration.T, aspect='auto', cmap='Reds',
                     extent=[0, N_SIMTIME, 100, 0])
    plt.colorbar(im2, ax=ax2, label='Gene 2 Concentration')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Position')
    ax2.set_title('Gene 2 Concentration', fontsize=12)
    
    # 3. Time series for specific cells
    ax3 = plt.subplot(2, 3, 3)
    cell_positions = [10, 30, 50, 70, 90]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_positions)))
    
    for i, pos in enumerate(cell_positions):
        ax3.plot(t, solution[:, pos, 0], color=colors[i], linestyle='-', 
                label=f'Gene 1, Cell {pos}', alpha=0.8)
        ax3.plot(t, solution[:, pos, 1], color=colors[i], linestyle='--', 
                label=f'Gene 2, Cell {pos}', alpha=0.8)
    
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Concentration')
    ax3.set_title('Time Series for Selected Cells')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase portrait for middle cell
    ax4 = plt.subplot(2, 3, 4)
    middle_cell = 50
    x1_middle = solution[:, middle_cell, 0]
    x2_middle = solution[:, middle_cell, 1]
    
    # Color points by time
    scatter = ax4.scatter(x1_middle, x2_middle, c=t, cmap='viridis', s=20, alpha=0.7)
    ax4.plot(x1_middle[0], x2_middle[0], 'go', markersize=8, label='Start')
    ax4.plot(x1_middle[-1], x2_middle[-1], 'ro', markersize=8, label='End')
    plt.colorbar(scatter, ax=ax4, label='Time')
    ax4.set_xlabel('Gene 1 Concentration')
    ax4.set_ylabel('Gene 2 Concentration')
    ax4.set_title(f'Phase Portrait (Cell {middle_cell})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Gradient function visualization
    ax5 = plt.subplot(2, 3, 5)
    time_points = [0, 20, 40, 60, 80]
    positions_plot = np.arange(N_CELLS)
    
    for t_val in time_points:
        g_values = np.minimum(np.exp(A * positions_plot - B * t_val), 1)
        ax5.plot(positions_plot, g_values, label=f't = {t_val}')
    
    ax5.set_xlabel('Cell Position')
    ax5.set_ylabel('Gradient g(i,t)')
    ax5.set_title('Gradient Function Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. System parameters summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    param_text = f"""
System Parameters:
Test State: {test_state}

Weight Matrix (scaled):
{W}

Diagonal Matrix (scaled):
{D}

D @ ones: {D_ONES}

Key Interactions:
• Gene 1 → Gene 1: +{W[0,0]:.1f} (self-activation)
• Gene 2 → Gene 1: {W[0,1]:.1f} (inhibition)
• Gene 1 → Gene 2: +{W[1,0]:.1f} (activation)
• Gene 2 → Gene 2: {W[1,1]:.1f} (no self-interaction)

Gradient Parameters:
A = {A:.4f}, B = {B:.4f}
g(i,t) = min(exp({A:.4f}*i - {B:.4f}*t), 1)

Reward: {reward}
"""
    
    ax6.text(0.05, 0.95, param_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('2node_somitogenesis_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return solution, t

def analyze_boundary_formation(test_state):
    """
    Analyze how boundaries form in the system.
    """
    print("\n" + "="*60)
    print("BOUNDARY FORMATION ANALYSIS")
    print("="*60)
    
    # Simulate with detailed output
    reward = somitogenesis_reward_func(test_state, plot=True)
    
    print(f"Final reward (boundaries detected): {reward}")
    
    # Additional analysis could be added here
    return reward

if __name__ == "__main__":
    # Run the test
    test_state, reward = test_2node_somitogenesis()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    solution, t = visualize_2node_system(test_state)
    
    # Analyze boundary formation
    analyze_boundary_formation(test_state)
    
    print(f"\nTest completed successfully!")
    print(f"Final reward: {reward}")
    print("Visualization saved as '2node_somitogenesis_analysis.png'") 
    
