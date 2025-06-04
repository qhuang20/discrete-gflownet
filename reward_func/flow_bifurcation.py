import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import fsolve
# from plot_somite_flow import plot_somite_flow


# Repressilator example
# test_state= [0, 0, -100, 0, 0, 0, -100, -80, 0, 100, 100, 100] # 3n Repressilator 
test_state=[70, 50, 10, -30,30, 20] # 2n somite-half
# test_state= [85, 50, 10, -80, 40, 20] # 2n somite-s
# test_state= [100, 0, 40, -100, 30, 20] # 2n somite-s
# test_state= [0, 90, 0, 50, 30, 20]  # 2n somite-1
# test_state=[100, 100, -60, 50, -75, 30] # 2n chaos

print("Testing state:")
print(f"State: {test_state}")
print(f"State length: {len(test_state)}")

# Calculate n_nodes to verify
n_nodes = int((-1 + (1 + 4*len(test_state))**0.5) / 2)
print(f"Number of nodes: {n_nodes}")

# Extract components
n_weights = n_nodes * n_nodes
weights = test_state[:n_weights]
d_values = test_state[n_weights:n_weights+n_nodes]

print(f"Weights: {weights}")
print(f"D values: {d_values}")




def find_fixed_points_and_stability(W, D_ONES, S, g_val, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5)):
    """
    Find fixed points and determine their stability using eigenvalue analysis.
    """
    from evo_devo import sigmoid
    
    def system_derivatives(vars):
        x1, x2 = vars
        x_current = np.array([x1, x2])
        z = g_val * D_ONES[:2] + x_current @ W[:2, :2].T
        sigmoid_z = sigmoid(z)
        decay = x_current * S[:2]
        return [sigmoid_z[0] - decay[0], sigmoid_z[1] - decay[1]]
    
    def jacobian_at_point(x1, x2):
        """Calculate Jacobian matrix at a given point"""
        x_current = np.array([x1, x2])
        z = g_val * D_ONES[:2] + x_current @ W[:2, :2].T
        
        # Derivative of sigmoid: sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z))
        sigmoid_z = sigmoid(z)
        sigmoid_prime = sigmoid_z * (1 - sigmoid_z)
        
        # Jacobian elements
        J = np.zeros((2, 2))
        J[0, 0] = sigmoid_prime[0] * W[0, 0] - S[0]
        J[0, 1] = sigmoid_prime[0] * W[0, 1]
        J[1, 0] = sigmoid_prime[1] * W[1, 0]
        J[1, 1] = sigmoid_prime[1] * W[1, 1] - S[1]
        
        return J
    
    # Find fixed points by searching from multiple initial guesses
    fixed_points = []
    initial_guesses = [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (0.5, 0.5), (0.2, 0.8), (0.8, 0.2),
        (x_range[0], y_range[0]), (x_range[1], y_range[1])
    ]
    
    for x0, y0 in initial_guesses:
        try:
            sol = fsolve(system_derivatives, [x0, y0], xtol=1e-12)
            # Check if it's actually a fixed point
            residual = system_derivatives(sol)
            if abs(residual[0]) < 1e-8 and abs(residual[1]) < 1e-8:
                # Check if it's within our range
                if x_range[0] <= sol[0] <= x_range[1] and y_range[0] <= sol[1] <= y_range[1]:
                    # Check if we already found this point
                    is_new = True
                    for existing_point, _ in fixed_points:
                        if abs(existing_point[0] - sol[0]) < 1e-6 and abs(existing_point[1] - sol[1]) < 1e-6:
                            is_new = False
                            break
                    
                    if is_new:
                        # Determine stability using eigenvalues
                        J = jacobian_at_point(sol[0], sol[1])
                        eigenvals = np.linalg.eigvals(J)
                        is_stable = all(np.real(eig) < 0 for eig in eigenvals)
                        fixed_points.append((sol, is_stable))
                        
        except:
            continue
    
    return fixed_points




def create_bifurcation_diagram(state, max_simtime=90, time_step=1, cell_position=99, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5)):
    """
    Create a bifurcation diagram showing how fixed points change with time/g parameter.
    
    Args:
        state: Complete state vector 
        max_simtime: Maximum simulation time
        time_step: Time step for parameter sweep
        cell_position: Which cell position to analyze (0 to N_CELLS-1)
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
    """
    # Parse the state vector
    n_nodes = int((-1 + (1 + 4*len(state))**0.5) / 2)
    n_weights = n_nodes * n_nodes
    weights = state[:n_weights]
    d_values = state[n_weights:n_weights+n_nodes]
    s_values = np.ones(n_nodes)
    
    # System parameters
    WEIGHT_SCALE = 10 
    DIAGONAL_SCALE = 10 
    
    # Get the weight matrix and diagonal matrix
    from evo_devo import weights_to_matrix
    W = weights_to_matrix(weights) / WEIGHT_SCALE
    D = np.diag(d_values) / DIAGONAL_SCALE
    D_ONES = D @ np.ones(n_nodes)
    
    # Fixed parameters
    A, B = 0.1/5, 0.2/5
    S = s_values
    
    # Create time array
    time_values = np.arange(0, max_simtime + time_step, time_step)
    
    # Store bifurcation data
    bifurcation_data = {
        'times': [],
        'g_values': [],
        'x1_stable': [],
        'x2_stable': [],
        'x1_unstable': [],
        'x2_unstable': []
    }
    
    print(f"\nComputing bifurcation diagram for cell position {cell_position} with {len(time_values)} parameter points...")
    
    # Suppress warnings for cleaner output
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for i, t in enumerate(time_values):
            if i % 10 == 0:  # Progress indicator
                print(f"Progress: {i}/{len(time_values)} ({100*i/len(time_values):.1f}%)")
            
            # Calculate g value
            g = np.minimum(np.exp(A * cell_position - B * t), 1)
            
            # Find fixed points at this parameter value
            fixed_points = find_fixed_points_and_stability(W, D_ONES, S, g, x_range, y_range)
            
            # Store data
            for (x_fp, y_fp), is_stable in fixed_points:
                bifurcation_data['times'].append(t)
                bifurcation_data['g_values'].append(g)
                
                if is_stable:
                    bifurcation_data['x1_stable'].append(x_fp)
                    bifurcation_data['x2_stable'].append(y_fp)
                    bifurcation_data['x1_unstable'].append(np.nan)
                    bifurcation_data['x2_unstable'].append(np.nan)
                else:
                    bifurcation_data['x1_stable'].append(np.nan)
                    bifurcation_data['x2_stable'].append(np.nan)
                    bifurcation_data['x1_unstable'].append(x_fp)
                    bifurcation_data['x2_unstable'].append(y_fp)
    
    # Create the bifurcation diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert to numpy arrays for plotting
    times = np.array(bifurcation_data['times'])
    g_vals = np.array(bifurcation_data['g_values'])
    x1_stable = np.array(bifurcation_data['x1_stable'])
    x2_stable = np.array(bifurcation_data['x2_stable'])
    x1_unstable = np.array(bifurcation_data['x1_unstable'])
    x2_unstable = np.array(bifurcation_data['x2_unstable'])
    
    # Plot 1: Gene 1 vs g parameter
    ax1.scatter(g_vals, x1_stable, c='green', s=8, alpha=0.6, label='Stable')
    ax1.scatter(g_vals, x1_unstable, c='red', s=8, alpha=0.6, label='Unstable')
    ax1.set_xlabel('g parameter')
    ax1.set_ylabel('Gene 1 concentration')
    ax1.set_title(f'Bifurcation Diagram: Gene 1 vs g (Cell pos {cell_position})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Gene 2 vs g parameter
    ax2.scatter(g_vals, x2_stable, c='green', s=8, alpha=0.6, label='Stable')
    ax2.scatter(g_vals, x2_unstable, c='red', s=8, alpha=0.6, label='Unstable')
    ax2.set_xlabel('g parameter')
    ax2.set_ylabel('Gene 2 concentration')
    ax2.set_title(f'Bifurcation Diagram: Gene 2 vs g (Cell pos {cell_position})')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'bifurcation_diagram_cell_{cell_position}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nBifurcation diagram completed for cell position {cell_position}!")
    print(f"Total fixed points found: {len(times)}")
    print(f"Stable points: {np.sum(~np.isnan(x1_stable))}")
    print(f"Unstable points: {np.sum(~np.isnan(x1_unstable))}")
    print(f"Bifurcation diagram saved as 'bifurcation_diagram_cell_{cell_position}.png'")
    
    return bifurcation_data

def create_flow_movie(state, max_simtime=90, time_step=5, cell_position=99, n_points=20, x_range=(-0.5, 1.5), y_range=(-0.5, 1.5)):
    """
    Create an animated movie of flow diagrams over time using streamlines.
    
    This function creates a phase space movie showing:
    1. Theoretical flow field (streamlines) - where the system WOULD go from any point
    2. Nullclines (red/blue lines) - where dx/dt = 0 for each gene
    3. Fixed points (green/red dots) - stable/unstable equilibrium points  
    4. Real trajectory (grey dots) - where the actual cell DOES go via ODE simulation
    
    The real trajectory is overlaid on the theoretical flow field by mapping between
    two different time grids using nearest neighbor interpolation.
    
    Args:
        state: Complete state vector 
        max_simtime: Maximum simulation time
        time_step: Time step between frames
        cell_position: Which cell position to analyze (0 to N_CELLS-1)
        n_points: Number of points in each direction for the flow field
        x_range: Tuple of (min, max) for x-axis
        y_range: Tuple of (min, max) for y-axis
    """
    # Parse the state vector
    n_nodes = int((-1 + (1 + 4*len(state))**0.5) / 2)
    n_weights = n_nodes * n_nodes
    weights = state[:n_weights]
    d_values = state[n_weights:n_weights+n_nodes]
    s_values = np.ones(n_nodes)
    
    # System parameters
    WEIGHT_SCALE = 10 
    DIAGONAL_SCALE = 10 
    
    # Get the weight matrix and diagonal matrix
    from evo_devo import weights_to_matrix, sigmoid, somitogenesis_sol_func
    W = weights_to_matrix(weights) / WEIGHT_SCALE
    D = np.diag(d_values) / DIAGONAL_SCALE
    D_ONES = D @ np.ones(n_nodes)
    
    # Fixed parameters
    A, B = 0.1/5, 0.2/5
    S = s_values
    
    # Get ODE solution using the reusable function
    print(f"Running ODE simulation for gene expression trajectory...")
    t_sim, cell_trajectory, full_solution = somitogenesis_sol_func(state, cell_position, max_simtime, plot=False)
    print(f"ODE simulation completed. Trajectory shape: {cell_trajectory.shape}")
    
    # Extract first two genes for phase space plot
    cell_trajectory_2d = cell_trajectory[:, :2]
    
    # Create time array for animation
    time_values = np.arange(0, max_simtime + time_step, time_step)
    
    # Create a grid of points for flow field
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        t = time_values[frame]
        
        # Initialize arrays for derivatives
        dX = np.zeros_like(X)
        dY = np.zeros_like(Y)
        
        # Calculate g value
        g = np.minimum(np.exp(A * cell_position - B * t), 1)
        
        # Calculate derivatives at each point
        for i in range(n_points):
            for j in range(n_points):
                # Create state vector with first two genes
                x_current = np.zeros(n_nodes)
                x_current[0] = X[i,j]
                x_current[1] = Y[i,j]
                
                # Calculate derivatives using the actual somitogenesis system dynamics
                z = g * D_ONES + x_current @ W.T
                sigmoid_z = sigmoid(z)
                decay = x_current * S
                
                # Store derivatives for first two genes
                dX[i,j] = sigmoid_z[0] - decay[0]
                dY[i,j] = sigmoid_z[1] - decay[1]
        
        # Create streamlines instead of quiver plot
        ax.streamplot(X, Y, dX, dY, color=np.sqrt(dX**2 + dY**2), cmap='viridis', 
                     density=1.5, linewidth=1, arrowsize=1.5)
        
        # Add nullclines
        nullcline_points = 50
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
                
                z = g * D_ONES + x_current @ W.T
                sigmoid_z = sigmoid(z)
                decay = x_current * S
                
                dX_fine[i,j] = sigmoid_z[0] - decay[0]
                dY_fine[i,j] = sigmoid_z[1] - decay[1]
        
        # Plot nullclines
        ax.contour(X_fine, Y_fine, dX_fine, levels=[0], colors='red', linewidths=2, linestyles='-', alpha=0.8)
        ax.contour(X_fine, Y_fine, dY_fine, levels=[0], colors='blue', linewidths=2, linestyles='-', alpha=0.8)
        
        # Find and plot fixed points
        fixed_points = find_fixed_points_and_stability(W, D_ONES, S, g, x_range, y_range)
        
        for (x_fp, y_fp), is_stable in fixed_points:
            color = 'green' if is_stable else 'red'
            ax.plot(x_fp, y_fp, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
        
        # Plot gene expression trajectory from ODE simulation
        # 
        # TIME MAPPING EXPLANATION:
        # - ODE simulation: t_sim has 200 points from 0 to 90 (high resolution)
        # - Animation frames: time_values has 19 points [0, 5, 10, 15, ..., 90] (low resolution)
        # - For each animation frame, we find the closest ODE simulation time point
        # - This allows us to show the actual gene expression trajectory on the flow field
        #
        # Example: Animation frame at t=25.0 maps to ODE index ~55 (t_sim[55] ≈ 25.0)
        
        # Find the closest time point in ODE simulation to current animation time
        t_idx = np.argmin(np.abs(t_sim - t))  # Nearest neighbor interpolation
        
        # Plot trajectory trail: show recent history of gene expression
        # This creates a "comet tail" effect showing where the cell has been
        n_trail_points = min(20, t_idx + 1)  # Show last 20 ODE points or up to current time
        start_idx = max(0, t_idx - n_trail_points + 1)
        
        if t_idx > 0:
            # Plot trajectory trail with fading alpha (older points fade out)
            for i in range(start_idx, t_idx):
                # Alpha increases from 0.0 to 0.3 as we approach current time
                alpha = (i - start_idx + 1) / n_trail_points * 0.3  # Fade from 0 to 0.3
                ax.plot(cell_trajectory_2d[i, 0], cell_trajectory_2d[i, 1], 'o', 
                       color='grey', markersize=3, alpha=alpha)
        
        # Plot current gene expression state as a prominent grey dot
        # This shows where the actual cell is RIGHT NOW in gene expression space
        if t_idx < len(cell_trajectory_2d):
            current_x1 = cell_trajectory_2d[t_idx, 0]  # Gene 1 concentration at current time
            current_x2 = cell_trajectory_2d[t_idx, 1]  # Gene 2 concentration at current time
            ax.plot(current_x1, current_x2, 'o', color='grey', markersize=10, 
                   markeredgecolor='black', markeredgewidth=2, alpha=0.8,
                   label=f'Cell {cell_position} state')
        
        # Set labels and title
        ax.set_xlabel('Gene 1 concentration')
        ax.set_ylabel('Gene 2 concentration')
        
        # Calculate g value for display
        g_val = g
        ax.set_title(f'Flow Diagram: t={t:.1f}, g={g_val:.3f} (Cell pos {cell_position})')
        ax.grid(True)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='dx₁/dt = 0'),
            Line2D([0], [0], color='blue', lw=2, label='dx₂/dt = 0'),
            Line2D([0], [0], marker='o', color='green', lw=0, markersize=8, 
                   markeredgecolor='black', label='Stable fixed point'),
            Line2D([0], [0], marker='o', color='red', lw=0, markersize=8, 
                   markeredgecolor='black', label='Unstable fixed point'),
            Line2D([0], [0], marker='o', color='grey', lw=0, markersize=10, 
                   markeredgecolor='black', label=f'Cell {cell_position} state')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(time_values), 
                                   interval=500, blit=False, repeat=True)
    
    # Save as gif
    print(f"\nCreating animation with {len(time_values)} frames for cell position {cell_position}...")
    print(f"Time range: 0 to {max_simtime} with step size {time_step}")
    
    # Save the animation
    anim.save(f'flow_dynamics_streamlines_cell_{cell_position}.gif', writer='pillow', fps=2)
    print(f"Animation saved as 'flow_dynamics_streamlines_cell_{cell_position}.gif'")
    
    # Also save as mp4 if ffmpeg is available
    try:
        anim.save(f'flow_dynamics_streamlines_cell_{cell_position}.mp4', writer='ffmpeg', fps=2)
        print(f"Animation also saved as 'flow_dynamics_streamlines_cell_{cell_position}.mp4'")
    except Exception as e:
        print(f"Could not save MP4 (ffmpeg might not be available): {e}")
    
    # Show the animation
    plt.show()
    
    return anim




cell_pos = 0  # change this to analyze different cell positions

# Create the movie
print("\nCreating flow dynamics movie with streamlines and fixed points...")
animation_obj = create_flow_movie(test_state, max_simtime=90, time_step=5, cell_position=cell_pos)

# Create the bifurcation diagram
print("\nCreating bifurcation diagram...")
bifurcation_data = create_bifurcation_diagram(test_state, max_simtime=90, time_step=1, cell_position=cell_pos)


