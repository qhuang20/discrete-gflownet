import os
import numpy as np
import matplotlib.pyplot as plt
import time
from graph.graph import draw_network_motif
from reward_func.evo_devo import coord_reward_func, oscillator_reward_func, somitogenesis_reward_func
import itertools
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm  # For progress tracking
import argparse

# Create output directory if it doesn't exist
os.makedirs("test_output", exist_ok=True)


# Top performing state from analysis
test_cases = [
    {"name": "top1", "state": [100, 85, -85, 55, -80, 50]},
    {"name": "top2", "state": [85, 50, 10, -80, 40, 20]},
    {"name": "top3", "state": [85, 50, 10, -80, 35, 20]}
]





def generate_plots(test_case):
    state = test_case["state"]
    name = test_case["name"]
    
    # Calculate number of nodes from the length of state vector
    n_nodes = int((-1 + (1 + 4*len(state))**0.5) / 2)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 12))
    
    # Draw network motif in second subplot
    draw_network_motif(state, ax=ax2)
    ax2.set_title(f"{n_nodes}-Node Network Motif")
    
    # Plot somite pattern and get reward
    start_time = time.perf_counter_ns()
    reward = somitogenesis_reward_func(state, plot=True, ax=ax1)
    end_time = time.perf_counter_ns()
    print(f"Reward for somitogenesis ({name}): {reward}")
    print(f"Time taken to run somitogenesis_reward_func: {(end_time - start_time)/1e9:.9f} seconds")
    
    # Add state to the title of the top plot
    ax1.set_title(f"Somite Pattern - State: {state}")
    
    plt.tight_layout()
    plt.savefig(f"test_output/network_plot_{name}.png")
    plt.close(fig)

def test_plot_generation():
    # Generate plots for each test case
    for test_case in test_cases:
        generate_plots(test_case)

    # Also generate a plot for a custom state configuration
    custom_state = [75, 45, 65, 100, -75, 30]  # Example from rank 9 in the analysis
    custom_case = {"name": "custom", "state": custom_state}
    generate_plots(custom_case)

    print("All plots saved to test_output/ directory")








def evaluate_state(args):
    """Evaluate a single state with specified suffix and return (state, reward)"""
    state_prefix, suffix = args
    # Add the suffix
    full_state = list(state_prefix) + list(suffix)
    reward = somitogenesis_reward_func(full_state, plot=False)  # No plotting for efficiency
    return (full_state, reward)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run grid search for network states')
    parser.add_argument('--suffix', type=int, nargs='+', default=[30, 20],
                        help='Suffix values to append to each state (default: [30, 20])')
    args = parser.parse_args()
    
    # Get the suffix from command line arguments
    suffix = args.suffix
    suffix_str = '_'.join(map(str, suffix))
    print(f"Using suffix: {suffix}")
    
    
    
    
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs("test_grid_search_results", exist_ok=True)
    
    # Create possible values for each dimension
    possible_values = list(range(-100, 101, 10))  # -100 to 100 with step 10
    
    # Calculate total number of states
    total_states = len(possible_values) ** 4
    print(f"Total number of states to evaluate: {total_states:,}")
    print(f"Estimated time with sequential processing: {total_states * 0.03 / 60:.2f} minutes")
    
    # Use multiprocessing to speed up computation
    num_workers = cpu_count()
    print(f"Using {num_workers} CPU cores for parallel processing")
    estimated_time = total_states * 0.03 / num_workers / 60
    print(f"Estimated time with parallel processing: {estimated_time:.2f} minutes")
    
    # Ask user if they want to proceed
    proceed = input(f"This will evaluate {total_states:,} states and take approximately {estimated_time:.2f} minutes. Proceed? (y/n) ")
    if proceed.lower() != 'y':
        print("Operation cancelled")
        return
    
    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        # Generate all possible combinations for the first 4 dimensions
        # prefixes = itertools.product(possible_values, repeat=4)
        args_list = [(prefix, suffix) for prefix in itertools.product(possible_values, repeat=4)]
        
        # Process states in parallel with progress bar
        results = list(tqdm(pool.imap(evaluate_state, args_list), total=total_states))
    
    # Sort results by reward (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Save all results with suffix in filename
    with open(f"test_grid_search_results/suffix_{suffix_str}_all_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save top 100 results in a more readable format
    top_100 = results[:100]
    with open(f"test_grid_search_results/suffix_{suffix_str}_top_100_results.txt", 'w') as f:
        f.write("Rank\tState\tReward\n")
        for i, (state, reward) in enumerate(top_100):
            f.write(f"{i+1}\t{state}\t{reward}\n")
    
    end_time = time.time()
    print(f"Grid search completed in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best state found: {results[0][0]} with reward: {results[0][1]}")
    print(f"All results saved to test_grid_search_results/suffix_{suffix_str}_all_results.pkl")
    print(f"Top 100 results saved to test_grid_search_results/suffix_{suffix_str}_top_100_results.txt")

if __name__ == "__main__":
    main()
    
    
