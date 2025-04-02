import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time




def coord_reward_func(state):
    # reward1 = sum(1 for coord in state if coord == 5) + 0.001 # args.min_reward
    reward1 = sum(1 for coord in state if coord == 6) 
    reward2 = sum(2 for coord in state if coord == 8) 
    return reward1 + reward2






# helpers for oscillator and somite

def sigmoid(z):
    """Sigmoid activation function with overflow protection"""
    # return 1 / (1 + np.exp(-np.clip(z, -500, 500))) 
    # print("z:", z)
    # exit()
    return 1 / (1 + np.exp(5 - np.clip(z, -500, 500))) 


def weights_to_matrix(weights, plot=False, ax=None):
    """
    Transform state vector to matrix following specific pattern:
    1 node: [w1] -> [[w1]]
    2 node: [w1,w2,w3,w4] -> [[w1,w4],[w3,w2]]
    3 node: [w1,w2,w3,w4,w5,w6,w7,w8,w9] -> [[w1,w4,w7],[w3,w2,w9],[w6,w8,w5]]
    
    The transformation preserves inner matrix structure when going from n to n+1 nodes,
    with new weights added in specific positions.
    
    Returns an integer weight matrix.
    """
    # Infer number of nodes from state length
    n_nodes = int(np.sqrt(len(weights)))
    assert n_nodes * n_nodes == len(weights), "State length must be a perfect square"
    
    if n_nodes == 1:
        return np.array([[int(weights[0])]], dtype=np.int32)
        
    # Initialize matrix with zeros
    W_matrix = np.zeros((n_nodes, n_nodes), dtype=np.int32)
    
    # Base case for 2x2
    if n_nodes == 2:
        W_matrix[0,0] = weights[0]  # w1
        W_matrix[1,1] = weights[1]  # w2
        W_matrix[1,0] = weights[2]  # w3
        W_matrix[0,1] = weights[3]  # w4
        return W_matrix
        
    # For larger matrices, first copy the (n-1)x(n-1) inner matrix
    prev_size = n_nodes - 1
    prev_matrix = weights_to_matrix(weights[:prev_size*prev_size])
    
    # Copy previous matrix to top-left corner
    W_matrix[:prev_size, :prev_size] = prev_matrix
    
    # Add new weights for nth node
    start_idx = prev_size * prev_size  # Start index for new weights
    
    # First place diagonal element
    W_matrix[n_nodes-1, n_nodes-1] = weights[start_idx]
    
    # Fill rest of last row and column alternating between row and column entries
    curr_idx = start_idx + 1
    for i in range(n_nodes-1):
        # Fill matrix entry W_x,i
        W_matrix[n_nodes-1, i] = weights[curr_idx]
        curr_idx += 1
        # Fill matrix entry W_i,x
        W_matrix[i, n_nodes-1] = weights[curr_idx]
        curr_idx += 1
        
    return W_matrix












def oscillator_reward_func(weights, plot=False):
    """
    Simulate n-node system with given weights and return reward.
    
    Args:
        weights: List of weights representing the weight matrix (n^2 elements for n nodes)
        plot: Boolean to control plotting
    
    Returns:
        float: Reward value based on number of sharp peaks that are not too damped
    """
    
    # Infer number of nodes from weights length
    n_nodes = int(np.sqrt(len(weights)))
    assert n_nodes * n_nodes == len(weights), "Weights length must be a perfect square"
    
    # System parameters for n-node system
    n_simtime = 60  
    n_timepoints = 600 
    delta_osc = 0.00002  # 0.0002
    rtol = 1e-4  # Relative tolerance for ODE solver
    atol = 1e-7  # Absolute tolerance for ODE solver
    x0 = np.array([0.1] * n_nodes)  # Initial conditions
    
    def n_node_system_with_sigmoid(t, x, W):
        """Define the dynamical system for the n-node system with sigmoid"""
        z = W.dot(x)  # Compute W * x
        sigmoid_z = sigmoid(z)  # Apply sigmoid
        dxdt = sigmoid_z - x  # Compute the derivative
        return dxdt
    
    def calculate_reward(sol, delta=delta_osc):
        """Calculate reward based on the number of sharp peaks in x1 that are not too damped"""
        x1 = sol.y[0]  # Focus only on x1
        dx1 = np.diff(x1)  # First derivative approximation
        peaks = 0
        peak_heights = []
        
        for j in range(1, len(dx1)):
            if dx1[j-1] > 0 and dx1[j] < 0:  # Detect a peak
                sharpness = x1[j] - (x1[j-1] + x1[j+1]) / 2
                if sharpness > delta_osc:  # Check if the peak is sharp
                    peak_heights.append(x1[j])
                    # For first peak, always count it
                    if len(peak_heights) == 1:
                        peaks += 1
                    # For subsequent peaks, check damping ratio
                    elif len(peak_heights) > 1:
                        damping_ratio = peak_heights[-1] / peak_heights[-2]
                        if damping_ratio > 0.99:  # Less than 1% damping 
                            peaks += 1
        return peaks

    # Simulate system
    t = np.linspace(0, n_simtime, n_timepoints)
    W = np.array(weights).reshape(n_nodes, n_nodes)
    
    sol = solve_ivp(n_node_system_with_sigmoid, (t[0], t[-1]), x0, t_eval=t,
                    method='RK45', rtol=rtol, atol=atol,
                    args=(W,))
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        for i in range(n_nodes):
            plt.plot(t, sol.y[i], label=f'$x_{i+1}$')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title(f'{n_nodes}-Node System Dynamics with Sigmoid')
        plt.legend()
        plt.grid(True)
        plt.show()

    return calculate_reward(sol)





def somitogenesis_reward_func(state, plot=False, ax=None):
    """
    Calculate reward based on gene expression pattern simulation.
    
    Args:
        state: 1D array containing weights (w), diagonal factors (d), and decay rates (s)
              For n nodes: n^2 weights + n d values + n s values = n^2 + 2n total parameters
        plot: bool, whether to plot the heatmap (default: False)
        ax: matplotlib axes object for plotting in a grid (default: None)
        
    Returns:
        float: Reward value based on pattern formation and stability
    """
    
    # n_nodes = int((-2 + (4 + 4*len(state))**0.5) / 2)  # solve quadratic: n^2 + 2n - len(state) = 0 
    n_nodes = int((-1 + (1 + 4*len(state))**0.5) / 2)  # solve quadratic: n^2 + n - len(state) = 0 
    n_weights = n_nodes * n_nodes
    weights = state[:n_weights]
    d_values = state[n_weights:n_weights+n_nodes]
    s_values = np.ones(n_nodes) # Generate s_values with all 1s 
    
    # System parameters - moved to constants for faster access
    N_CELLS = 100
    N_SIMTIME = 90
    N_TIMEPOINTS = 200
    DELTA_SOMITE = 0.1
    DELTA_STABILITY = 0.02
    STABILITY_WEIGHT = 1.0
    SPARSITY_WEIGHT = 0.8   # 0 - 0.8 (max) 
    STABILITY_POWER = 5  # smaller tolerates waves more 
    N_BOUNDARY_CHECKS = 3
    RTOL = 1e-3
    ATOL = 1e-6
    # WEIGHT_SCALE = 10 
    # WEIGHT_SCALE = 5 
    WEIGHT_SCALE = 10 
    DIAGONAL_SCALE = 10 
    
    
    # Pre-compute initial conditions
    x0 = np.full(N_CELLS * n_nodes, 0.1)  # Faster than tile
    
    # Fixed parameters
    D = np.diag(d_values) / DIAGONAL_SCALE  # Scale diagonal values
    D_ONES = D @ np.ones(n_nodes)
    # A, B = 0.1, 0.2
    A, B = 0.1/2.5, 0.2/2.5 
    S = s_values  # Decay rates
    
    # Pre-compute positions array
    positions = np.arange(N_CELLS).reshape(-1, 1)



    def n_node_system(t, x, W):
        """System dynamics with node-specific parameters"""
        x_reshaped = x.reshape(-1, n_nodes)
        g = np.minimum(np.exp(A * positions - B * t), 1)
        z = g * D_ONES + x_reshaped @ W.T
        sigmoid_z = sigmoid(z)  # Use the protected sigmoid function instead
        # sigmoid_z = 1 / (1 + np.exp(-z)) 
        decay = x_reshaped * S  # S already has shape (n_nodes,) which broadcasts correctly with (N_CELLS, n_nodes)
        # print(W)  # Print the weight matrix
        return (sigmoid_z - decay).flatten()

    def simulate_system(weights):
        """Simulate with optimized matrix operations"""
        W = weights_to_matrix(weights) / WEIGHT_SCALE     
        t = np.linspace(0, N_SIMTIME, N_TIMEPOINTS)
        
        sol = solve_ivp(
            lambda t, x: n_node_system(t, x, W),
            (t[0], t[-1]),
            x0,
            t_eval=t,
            method='RK45',
            rtol=RTOL,
            atol=ATOL
        )
        return t, sol.y.T.reshape(len(t), N_CELLS, n_nodes)

    def count_boundaries(concentrations):
        """Count boundaries with minimum distance between them"""
        n_boundaries = 0
        last_boundary_pos = -float('inf')  # Position of last detected boundary
        
        for i in range(len(concentrations)-1):
            diff = abs(concentrations[i+1] - concentrations[i])
            if diff > DELTA_SOMITE:
                # Only count if far enough from last boundary (minimum 4 cells apart)
                if i - last_boundary_pos >= 4: 
                    n_boundaries += 1
                    last_boundary_pos = i
        return n_boundaries

    def sparsity_reward_combined(state, w1=0.0, w2=1.0):
        # Entropy-based component
        # Normalize values to probabilities
        abs_values = np.abs(state)
        if sum(abs_values) == 0:
            entropy_reward = 1.0  # maximum sparsity
        else:
            probs = abs_values / sum(abs_values)
            # Calculate entropy (lower entropy = more sparse)
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            entropy_reward = 1 / (1 + entropy)  # transform to reward
        
        # L0 component (explicitly rewards zeros)
        n_zeros = sum(1 for x in state if x == 0)
        l0_reward = n_zeros / len(state)
        
        return w1 * entropy_reward + w2 * l0_reward

    def calculate_reward(x1_concentration):
        """Optimized reward calculation"""
        mid_idx = len(x1_concentration) // 2
        check_indices = np.linspace(mid_idx, len(x1_concentration)-1, N_BOUNDARY_CHECKS, dtype=int)
        
        total_boundaries = sum(count_boundaries(x1_concentration[idx]) for idx in check_indices)
        if plot: print(f"Total boundaries across {N_BOUNDARY_CHECKS} timepoints: {total_boundaries}")
        
        # Calculate sparsity reward - always available
        sparsity_factor = round(1.0 + (SPARSITY_WEIGHT * sparsity_reward_combined(weights)), 3)
        if plot: print(f"Sparsity factor: {sparsity_factor}")
        
        if total_boundaries <= 2:  # to prevent slope 
            return sparsity_factor - 1.0  # Return just the sparsity component when boundaries are insufficient
            
        # Vectorized stability calculation
        second_half = x1_concentration[mid_idx:]
        changes = np.abs(np.diff(second_half, axis=0)) > DELTA_STABILITY
        total_changes = np.sum(changes)
        max_possible_changes = (len(second_half) - 1) * N_CELLS
        
        stability_reward = round((1 - total_changes / max_possible_changes) ** STABILITY_POWER, 3)
        if plot: print(f"Stability reward: {stability_reward}")
        
        # Multiplicative sparsity reward
        return round(total_boundaries * stability_reward * sparsity_factor, 3) 

    def plot_heatmap(x1_concentration, t, ax=None):
        """Plot heatmap (unchanged since plotting is not performance critical)"""
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
            
        im = ax.imshow(x1_concentration.T, aspect='auto', cmap='Blues',
                      extent=[0, N_SIMTIME, 100, 0])
        plt.colorbar(im, ax=ax, label='x1 Concentration')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title('x1 Concentration Across Time and Space')
        
        if ax is None:
            plt.show()
    # Main execution
    t, sol = simulate_system(weights)
    x1_concentration = sol[:, :, 0]
    # x1_concentration = sol[:, :, 2] 
    
    # Print the last 5 values of the first cell for x1, x2, x3
    if plot:
        # print("Last 5 values of first cell:")
        # print(f"x1: {sol[-5:, 0, 0]}")
        # print(f"x2: {sol[-5:, 0, 1]}")
        # print(f"x3: {sol[-5:, 0, 2]}")
        plot_heatmap(x1_concentration, t, ax)
    
    return calculate_reward(x1_concentration)




