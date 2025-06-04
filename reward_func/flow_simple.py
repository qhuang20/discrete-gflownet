import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from evo_devo import sigmoid, weights_to_matrix

def plot_somite_flow(state, n_points=20, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5), show_nullclines=False, t=0):
    """
    Plot the flow diagram for the first two genes in the somitogenesis system.
    
    Args:
        state: Complete state vector containing weights (w), diagonal factors (d), and decay rates (s)
               For n nodes: n^2 weights + n d values + n s values = n^2 + 2n total parameters
        n_points: Number of points in each direction for the flow field
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
        show_nullclines: Boolean flag to show nullclines (dx1/dt=0 and dx2/dt=0)
        t: Time parameter for the gradient term g = min(exp(A*position - B*t), 1)
    """
    # Parse the state vector like in somitogenesis_reward_func
    n_nodes = int((-1 + (1 + 4*len(state))**0.5) / 2)  # solve quadratic: n^2 + n - len(state) = 0 
    n_weights = n_nodes * n_nodes
    weights = state[:n_weights]
    d_values = state[n_weights:n_weights+n_nodes]
    s_values = np.ones(n_nodes)  # Generate s_values with all 1s 
    
    # System parameters from original function
    WEIGHT_SCALE = 10 
    DIAGONAL_SCALE = 10 
    
    # Get the weight matrix and diagonal matrix
    W = weights_to_matrix(weights) / WEIGHT_SCALE
    D = np.diag(d_values) / DIAGONAL_SCALE
    D_ONES = D @ np.ones(n_nodes)
    
    # Fixed parameters from original
    A, B = 0.1/5, 0.2/5
    S = s_values  # Decay rates
    
    # Position parameter - using last cell position
    N_CELLS = 100  # From original somitogenesis function
    last_cell_position = N_CELLS - 1  # Position 99 (last cell)
    
    # Create a grid of points
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Initialize arrays for derivatives
    dX = np.zeros_like(X)
    dY = np.zeros_like(Y)
    
    # Calculate derivatives at each point
    for i in range(n_points):
        for j in range(n_points):
            # Create state vector with first two genes
            x_current = np.zeros(n_nodes)
            x_current[0] = X[i,j]
            x_current[1] = Y[i,j]
            
            # Calculate derivatives using the actual somitogenesis system dynamics
            # Note: For flow diagram, we use last cell position and variable time t
            g = np.minimum(np.exp(A * last_cell_position - B * t), 1)  # g at position=last cell, time=t
            z = g * D_ONES + x_current @ W.T
            sigmoid_z = sigmoid(z)
            decay = x_current * S
            
            # Store derivatives for first two genes
            dX[i,j] = sigmoid_z[0] - decay[0]
            dY[i,j] = sigmoid_z[1] - decay[1]
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, dX, dY, np.sqrt(dX**2 + dY**2), cmap='viridis')
    plt.colorbar(label='Flow magnitude')
    plt.xlabel('Gene 1 concentration')
    plt.ylabel('Gene 2 concentration')
    plt.title(f'Flow Diagram for First Two Genes (t={t:.1f}, pos={last_cell_position})')
    plt.grid(True)
    
    # Add nullclines if requested
    if show_nullclines:
        # Create a finer grid for nullclines
        nullcline_points = 100
        x_fine = np.linspace(x_range[0], x_range[1], nullcline_points)
        y_fine = np.linspace(y_range[0], y_range[1], nullcline_points)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        
        # Calculate derivatives on fine grid
        dX_fine = np.zeros_like(X_fine)
        dY_fine = np.zeros_like(Y_fine)
        
        for i in range(nullcline_points):
            for j in range(nullcline_points):
                x_current = np.zeros(n_nodes)
                x_current[0] = X_fine[i,j]
                x_current[1] = Y_fine[i,j]
                
                g = np.minimum(np.exp(A * last_cell_position - B * t), 1)
                z = g * D_ONES + x_current @ W.T
                sigmoid_z = sigmoid(z)
                decay = x_current * S
                
                dX_fine[i,j] = sigmoid_z[0] - decay[0]
                dY_fine[i,j] = sigmoid_z[1] - decay[1]
        
        # Plot nullclines using contour lines
        plt.contour(X_fine, Y_fine, dX_fine, levels=[0], colors='red', linewidths=2, linestyles='-', alpha=0.8)
        plt.contour(X_fine, Y_fine, dY_fine, levels=[0], colors='blue', linewidths=2, linestyles='-', alpha=0.8)
        
        # Add legend for nullclines
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='red', lw=2, label='dx₁/dt = 0'),
                          Line2D([0], [0], color='blue', lw=2, label='dx₂/dt = 0')]
        plt.legend(handles=legend_elements, loc='upper right')    
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example state for a 3-node system: 9 weights + 3 d_values = 12 parameters
    example_state = [1, 0, 0, 0, 1, 0, 0, 0, 1,  # 3x3 identity weight matrix
                     1, 1, 1]  # d_values
    plot_somite_flow(example_state) 
    
    