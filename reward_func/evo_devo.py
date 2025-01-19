import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time




def coord_reward_func(state):
    # reward1 = sum(1 for coord in state if coord == 5) + 0.001 # args.min_reward
    reward1 = sum(1 for coord in state if coord == 6) 
    reward2 = sum(2 for coord in state if coord == 8) 
    return reward1 + reward2




def sigmoid(z):
    """Sigmoid activation function with overflow protection"""
    return 1 / (1 + np.exp(-np.clip(z, -5000, 5000)))





def oscillator_reward_func(weights, plot=False):
    """
    Simulate 3-node system with given weights and return reward.
    
    Args:
        weights: List of 9 weights [w11, w12, w13, w21, w22, w23, w31, w32, w33]
        plot: Boolean to control plotting
    
    Returns:
        float: Reward value based on number of sharp peaks that are not too damped
    """
    
    # System parameters for 3-node system
    n_simtime = 60  
    n_timepoints = 600 
    delta_osc = 0.00002  # 0.0002
    rtol = 1e-4  # Relative tolerance for ODE solver
    atol = 1e-7  # Absolute tolerance for ODE solver
    x0 = np.array([0.1, 0.1, 0.1])  # Initial conditions
    
    def three_node_system_with_sigmoid(t, x, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        """Define the dynamical system for the 3-node system with sigmoid"""
        M_tilde = np.array([[w11, w12, w13],
                           [w21, w22, w23],
                           [w31, w32, w33]])
        
        z = M_tilde.dot(x)  # Compute M_tilde * x
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
    w11, w12, w13, w21, w22, w23, w31, w32, w33 = weights
    
    sol = solve_ivp(three_node_system_with_sigmoid, (t[0], t[-1]), x0, t_eval=t,
                    method='RK45', rtol=rtol, atol=atol,
                    args=(w11, w12, w13, w21, w22, w23, w31, w32, w33))
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(t, sol.y[0], label='$x_1$')
        plt.plot(t, sol.y[1], label='$x_2$')
        plt.plot(t, sol.y[2], label='$x_3$')
        plt.xlabel('Time')
        plt.ylabel('Concentration')
        plt.title('3-Node System Dynamics with Sigmoid')
        plt.legend()
        plt.grid(True)
        plt.show()

    return calculate_reward(sol)




# def somitogenesis_reward_func(state, plot=False):
#     """
#     Calculate reward based on gene expression pattern simulation.
    
#     Args:
#         state: 1D array of weights [w11, w12, w13, w21, w22, w23, w31, w32, w33]
#         plot: bool, whether to plot the heatmap (default: False)
        
#     Returns:
#         float: Reward value based on pattern formation and stability
#     """
#     # System parameters for 3-node system
#     n_cells = 100  # Number of cells/positions
#     n_simtime = 60  # Total simulation time
#     n_timepoints = 200 
#     delta_somite = 0.1
#     delta_stable = 0.01  
#     rtol = 1e-4  # Relative tolerance for ODE solver
#     atol = 1e-7  # Absolute tolerance for ODE solver
#     weight_scale = 10.0  # Scaling factor for weights
#     x0_sc = np.array([0.1, 0.1, 0.1]) # initial condition for single cell
#     x0 = np.tile(x0_sc, n_cells) # initial condition for all cells 
    
#     # Other fixed parameters
#     d1, d2, d3 = 1, 1, 1
#     a, b = 0.1, 0.2


#     def three_node_system(t, x, w11, w12, w13, w21, w22, w23, w31, w32, w33, d1, d2, d3, a, b):
#         """Define the dynamical system for the 3-node network"""
#         x_reshaped = x.reshape(-1, 3)
        
#         W = np.array([[w11/weight_scale, w12/weight_scale, w13/weight_scale],
#                       [w21/weight_scale, w22/weight_scale, w23/weight_scale],
#                       [w31/weight_scale, w32/weight_scale, w33/weight_scale]])
        
#         positions = np.arange(n_cells)
#         g = np.minimum(np.exp(a * positions - b * t), 1)
#         g = g.reshape(-1, 1)
        
#         D = np.array([[d1, 0, 0],
#                       [0, d2, 0], 
#                       [0, 0, d3]])
#         D_ones = D @ np.ones(3)
        
#         z = g * D_ones + x_reshaped @ W.T
#         dxdt = sigmoid(z) - x_reshaped
        
#         return dxdt.flatten()

#     def simulate_system(w11, w12, w13, w21, w22, w23, w31, w32, w33):
#         """Simulate the system across time and space"""
#         t = np.linspace(0, n_simtime, n_timepoints)
        
#         sol = solve_ivp(three_node_system, (t[0], t[-1]), x0, t_eval=t, 
#                        method='RK45', rtol=rtol, atol=atol,
#                        args=(w11, w12, w13, w21, w22, w23, w31, w32, w33, d1, d2, d3, a, b))
        
#         return t, sol.y.T.reshape(len(t), n_cells, 3)

#     def calculate_reward(x1_concentration, delta_somite=delta_somite):
#         """Calculate reward based on changes in x1 concentration and stability"""
#         total_reward = 0
        
#         # Spatial pattern reward
#         spatial_reward = 0
#         final_concentrations = x1_concentration[-1]
#         for i in range(len(final_concentrations)-1):
#             if abs(final_concentrations[i+1] - final_concentrations[i]) > delta_somite:
#                 spatial_reward += 1
#         # print(f"spatial_reward: {spatial_reward}")
        
#         # Only check stability if spatial pattern exists
#         stability_reward = 0
#         if spatial_reward > 0:
#             last_5_timesteps = x1_concentration[-5:]
#             for cell in range(n_cells):
#                 cell_concentrations = last_5_timesteps[:, cell]
#                 is_stable = True
#                 for t in range(len(cell_concentrations)-1):
#                     if abs(cell_concentrations[t+1] - cell_concentrations[t]) > delta_stable: 
#                         is_stable = False
#                         break
#                 if is_stable:
#                     stability_reward += 1
#             # print(f"stability_reward: {stability_reward}")
        
#         total_reward = spatial_reward * (stability_reward / n_cells)  
#         return total_reward

#     def plot_heatmap(x1_concentration, t):
#         """Plot heatmap of x1 concentration across time and space"""
#         plt.figure(figsize=(10, 6))
#         plt.imshow(x1_concentration.T, aspect='auto', cmap='Blues',
#                   extent=[0, n_simtime, 100, 0])
#         plt.colorbar(label='x1 Concentration')
#         plt.xlabel('Time')
#         plt.ylabel('Position')
#         plt.title('x1 Concentration Across Time and Space')
#         plt.show()
        
#         # print("x1 concentration at last time step:")
#         # print(np.array2string(x1_concentration[-1], precision=3, suppress_small=True, floatmode='fixed')) 

#     # Run simulation
#     w11, w12, w13, w21, w22, w23, w31, w32, w33 = state
#     t, sol = simulate_system(w11, w12, w13, w21, w22, w23, w31, w32, w33)
#     x1_concentration = sol[:, :, 0]  # Extract x1 concentrations
    
#     if plot:
#         plot_heatmap(x1_concentration, t)
    
#     return calculate_reward(x1_concentration)





# def somitogenesis_reward_func(state, plot=False, subplot=None):
#     """
#     Calculate reward based on gene expression pattern simulation.
    
#     Args:
#         state: 1D array of weights [w11, w12, w13, w21, w22, w23, w31, w32, w33]
#         plot: bool, whether to plot the heatmap (default: False)
#         subplot: matplotlib subplot object for plotting in a grid (default: None)
        
#     Returns:
#         float: Reward value based on pattern formation and stability
#     """
#     # System parameters for 3-node system
#     n_cells = 100  # Number of cells/positions
#     n_simtime = 60  # Total simulation time
#     n_timepoints = 200 
#     delta_somite = 0.038 # 0.1
#     delta_stability = 0.02
#     stability_weight = 1.0  # Weight for stability reward
#     rtol = 1e-3  # Relative tolerance for ODE solver
#     atol = 1e-6  # Absolute tolerance for ODE solver
#     weight_scale = 10.0  # Scaling factor for weights
#     n_boundary_checks = 3  # Number of times to check boundaries in second half
#     x0_sc = np.array([0.1, 0.1, 0.1]) # initial condition for single cell
#     x0 = np.tile(x0_sc, n_cells) # initial condition for all cells 
    
#     # Other fixed parameters
#     d1, d2, d3 = 1, 1, 1
#     a, b = 0.1, 0.2

#     def three_node_system(t, x, w11, w12, w13, w21, w22, w23, w31, w32, w33, d1, d2, d3, a, b):
#         """Define the dynamical system for the 3-node network"""
#         x_reshaped = x.reshape(-1, 3)
        
#         W = np.array([[w11/weight_scale, w12/weight_scale, w13/weight_scale],
#                       [w21/weight_scale, w22/weight_scale, w23/weight_scale],
#                       [w31/weight_scale, w32/weight_scale, w33/weight_scale]])
        
#         positions = np.arange(n_cells)
#         g = np.minimum(np.exp(a * positions - b * t), 1)
#         g = g.reshape(-1, 1)
        
#         D = np.array([[d1, 0, 0],
#                       [0, d2, 0], 
#                       [0, 0, d3]])
#         D_ones = D @ np.ones(3)
        
#         z = g * D_ones + x_reshaped @ W.T
#         dxdt = sigmoid(z) - x_reshaped
        
#         return dxdt.flatten()

#     def simulate_system(w11, w12, w13, w21, w22, w23, w31, w32, w33):
#         """Simulate the system across time and space"""
#         t = np.linspace(0, n_simtime, n_timepoints)
        
#         sol = solve_ivp(three_node_system, (t[0], t[-1]), x0, t_eval=t, 
#                        method='RK45', rtol=rtol, atol=atol,
#                        args=(w11, w12, w13, w21, w22, w23, w31, w32, w33, d1, d2, d3, a, b))
        
#         return t, sol.y.T.reshape(len(t), n_cells, 3)
    
#     def calculate_reward(x1_concentration, delta_somite=delta_somite):
#         """Calculate reward based on changes in x1 concentration and stability"""
#         # Get indices for second half of simulation
#         mid_idx = len(x1_concentration) // 2
#         check_indices = np.linspace(mid_idx, len(x1_concentration)-1, n_boundary_checks, dtype=int)
        
#         # Count boundaries at multiple timepoints
#         total_boundaries = 0
#         for idx in check_indices:
#             n_boundaries = 0
#             concentrations = x1_concentration[idx]
#             for i in range(len(concentrations)-1):
#                 if abs(concentrations[i+1] - concentrations[i]) > delta_somite:
#                     n_boundaries += 1
#             total_boundaries += n_boundaries
#             # print(f"Boundaries at timepoint {idx}: {n_boundaries}")
#         # print(f"Total boundaries across {n_boundary_checks} timepoints: {total_boundaries}")
        
#         # Calculate stability reward based on concentration changes for each cell
#         # in the second half of simulation
#         second_half = x1_concentration[mid_idx:, :]
#         stability_reward = 0
        
#         if total_boundaries > 2:  # Only add stability reward if boundaries > 2
#             max_possible_changes = len(second_half) - 1  # Maximum possible changes per cell
#             total_possible_changes = max_possible_changes * n_cells
            
#             total_changes = 0
#             # For each cell position
#             for cell_idx in range(second_half.shape[1]):
#                 cell_concentrations = second_half[:, cell_idx]
#                 concentration_changes = 0
#                 # Count significant changes in concentration over time
#                 for t in range(1, len(cell_concentrations)):
#                     if abs(cell_concentrations[t] - cell_concentrations[t-1]) > delta_stability:
#                         concentration_changes += 1
#                 total_changes += concentration_changes
                
#             # Convert changes to stability reward (fewer changes = higher reward)
#             stability_reward = round(stability_weight * (1 - total_changes / total_possible_changes), 3)
#             # print(f"Stability reward: {stability_reward}")
        
#         # Multiply total_boundaries by (1 + stability_reward) to create a boosting effect
#         # When stability_reward is positive, it will enhance the boundary reward
#         total_reward = round(total_boundaries * (stability_reward ** 10), 3)   
#         # print(f"total_reward: {total_reward}") 
#         return total_reward

#     def plot_heatmap(x1_concentration, t, subplot=None):
#         """Plot heatmap of x1 concentration across time and space"""
#         if subplot is None:
#             plt.figure(figsize=(10, 6))
#             ax = plt.gca()
#         else:
#             ax = subplot
            
#         im = ax.imshow(x1_concentration.T, aspect='auto', cmap='Blues',
#                       extent=[0, n_simtime, 100, 0])
#         plt.colorbar(im, ax=ax, label='x1 Concentration')
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Position')
#         ax.set_title('x1 Concentration Across Time and Space')
        
#         if subplot is None:
#             plt.show()

#     # Run simulation
#     w11, w12, w13, w21, w22, w23, w31, w32, w33 = state
#     t, sol = simulate_system(w11, w12, w13, w21, w22, w23, w31, w32, w33)
#     x1_concentration = sol[:, :, 0]  # Extract x1 concentrations
    
#     if plot:
#         plot_heatmap(x1_concentration, t, subplot)
    
#     return calculate_reward(x1_concentration)




def somitogenesis_reward_func(state, plot=False, subplot=None):
    """
    Calculate reward based on gene expression pattern simulation.
    
    Args:
        state: 1D array of weights [w11, w12, w13, w21, w22, w23, w31, w32, w33]
        plot: bool, whether to plot the heatmap (default: False)
        subplot: matplotlib subplot object for plotting in a grid (default: None)
        
    Returns:
        float: Reward value based on pattern formation and stability
    """
    
    # System parameters - moved to constants for faster access
    N_CELLS = 100
    N_SIMTIME = 90
    N_TIMEPOINTS = 200
    DELTA_SOMITE = 0.1
    DELTA_STABILITY = 0.02
    STABILITY_WEIGHT = 1.0
    STABILITY_POWER = 10
    RTOL = 1e-3
    ATOL = 1e-6
    WEIGHT_SCALE = 20
    N_BOUNDARY_CHECKS = 3
    
    # Pre-compute initial conditions
    x0 = np.full(N_CELLS * 3, 0.1)  # Faster than tile
    
    # Fixed parameters
    D = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
    D_ONES = D @ np.ones(3)
    A, B = 0.1, 0.2
    
    # Pre-compute positions array
    positions = np.arange(N_CELLS).reshape(-1, 1)

    def three_node_system(t, x, W):
        """System dynamics"""
        x_reshaped = x.reshape(-1, 3)
        g = np.minimum(np.exp(A * positions - B * t), 1)
        z = g * D_ONES + x_reshaped @ W.T
        return (1 / (1 + np.exp(-z)) - x_reshaped).flatten()

    def simulate_system(weights):
        """Simulate with optimized matrix operations"""
        W = np.array(weights).reshape(3,3) / WEIGHT_SCALE
        t = np.linspace(0, N_SIMTIME, N_TIMEPOINTS)
        
        sol = solve_ivp(
            lambda t, x: three_node_system(t, x, W),
            (t[0], t[-1]),
            x0,
            t_eval=t,
            method='RK45',
            rtol=RTOL,
            atol=ATOL
        )
        return t, sol.y.T.reshape(len(t), N_CELLS, 3)

    def count_boundaries(concentrations):
        """Count boundaries"""
        n_boundaries = 0
        for i in range(len(concentrations)-1):
            if abs(concentrations[i+1] - concentrations[i]) > DELTA_SOMITE:
                n_boundaries += 1
        return n_boundaries

    def calculate_reward(x1_concentration):
        """Optimized reward calculation"""
        mid_idx = len(x1_concentration) // 2
        check_indices = np.linspace(mid_idx, len(x1_concentration)-1, N_BOUNDARY_CHECKS, dtype=int)
        
        total_boundaries = sum(count_boundaries(x1_concentration[idx]) for idx in check_indices)
        if plot: print(f"Total boundaries across {N_BOUNDARY_CHECKS} timepoints: {total_boundaries}")
        
        if total_boundaries <= 3:
            return 0.0
            
        # Vectorized stability calculation
        second_half = x1_concentration[mid_idx:]
        changes = np.abs(np.diff(second_half, axis=0)) > DELTA_STABILITY
        total_changes = np.sum(changes)
        max_possible_changes = (len(second_half) - 1) * N_CELLS
        
        stability_reward = round(STABILITY_WEIGHT * (1 - total_changes / max_possible_changes), 3)
        if plot: print(f"Stability reward: {stability_reward}")
        
        return round(total_boundaries * (stability_reward ** STABILITY_POWER), 3)

    def plot_heatmap(x1_concentration, t, subplot=None):
        """Plot heatmap (unchanged since plotting is not performance critical)"""
        if subplot is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()
        else:
            ax = subplot
            
        im = ax.imshow(x1_concentration.T, aspect='auto', cmap='Blues',
                      extent=[0, N_SIMTIME, 100, 0])
        plt.colorbar(im, ax=ax, label='x1 Concentration')
        ax.set_xlabel('Time')
        ax.set_ylabel('Position')
        ax.set_title('x1 Concentration Across Time and Space')
        
        if subplot is None:
            plt.show()

    # Main execution
    t, sol = simulate_system(state)
    x1_concentration = sol[:, :, 0]
    
    if plot:
        plot_heatmap(x1_concentration, t, subplot)
    
    return calculate_reward(x1_concentration)




