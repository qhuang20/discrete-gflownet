import os
import numpy as np
import matplotlib.pyplot as plt
import time
from evo_devo import coord_reward_func, oscillator_reward_func, somitogenesis_reward_func
import itertools
from multiprocessing import Pool, cpu_count
import pickle
from tqdm import tqdm  # For progress tracking
import argparse


def evaluate_state(state):
    """Evaluate a single state and return (state, reward)"""
    reward = somitogenesis_reward_func(state, plot=False)  # No plotting for efficiency
    return (state, reward)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run grid search for network states')
    parser.add_argument('--dimensions', type=int, default=6,
                        help='Number of dimensions for the state vector (default: 6)')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt and proceed automatically')
    args = parser.parse_args()
    
    # Get the number of dimensions from command line arguments
    dimensions = args.dimensions
    print(f"Running grid search for {dimensions}-dimensional state vectors")
    
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs("reward_landscape_grid_search_results", exist_ok=True)
    
    # Create possible values for each dimension
    possible_values = list(range(-100, 101, 10))  # -100 to 100 with step 10
    
    # Calculate total number of states for ALL dimensions
    total_states = len(possible_values) ** dimensions
    print(f"Total number of states to evaluate: {total_states:,}")
    print(f"Estimated time with sequential processing: {total_states * 0.03 / 60:.2f} minutes")
    
    # Use multiprocessing to speed up computation
    total_cores = cpu_count()
    # num_workers = total_cores // 2  # Use half of available cores
    num_workers = 20
    print(f"Total CPU cores available: {total_cores}")
    print(f"Using {num_workers} CPU cores for parallel processing (half of available cores)")
    estimated_time = total_states * 0.03 / num_workers / 60
    print(f"Estimated time with parallel processing: {estimated_time:.2f} minutes")
    
    # Ask user if they want to proceed (unless --yes flag is used)
    if not args.yes:
        proceed = input(f"This will evaluate {total_states:,} states and take approximately {estimated_time:.2f} minutes. Proceed? (y/n) ")
        if proceed.lower() != 'y':
            print("Operation cancelled")
            return
    else:
        print(f"Auto-confirming: Will evaluate {total_states:,} states and take approximately {estimated_time:.2f} minutes.")
    
    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        # Generate all possible combinations for ALL dimensions
        all_states = [list(state) for state in itertools.product(possible_values, repeat=dimensions)]
        
        # Process states in parallel with progress bar
        results = list(tqdm(pool.imap(evaluate_state, all_states), total=total_states))
    
    # Sort results by reward (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Save all results
    with open(f"reward_landscape_grid_search_results/full_{dimensions}d_all_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    # Save top 100 results in a more readable format
    top_100 = results[:100]
    with open(f"reward_landscape_grid_search_results/full_{dimensions}d_top_100_results.txt", 'w') as f:
        f.write("Rank\tState\tReward\n")
        for i, (state, reward) in enumerate(top_100):
            f.write(f"{i+1}\t{state}\t{reward}\n")
    
    end_time = time.time()
    print(f"Grid search completed in {(end_time - start_time) / 60:.2f} minutes")
    print(f"Best state found: {results[0][0]} with reward: {results[0][1]}")
    print(f"All results saved to reward_landscape_grid_search_results/full_{dimensions}d_all_results.pkl")
    print(f"Top 100 results saved to reward_landscape_grid_search_results/full_{dimensions}d_top_100_results.txt")

if __name__ == "__main__":
    main()
    
    
