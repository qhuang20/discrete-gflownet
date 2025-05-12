import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import umap
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm
import time
import argparse

def analyze_results():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze grid search results with dimensionality reduction')
    parser.add_argument('--reward-threshold', type=float, default=0.0, 
                        help='Show only rewards greater than this threshold')
    
    parser.add_argument('--output-dir', type=str, default="analysis_output",
                        help='Directory name for output files (default: analysis_output)')
    parser.add_argument('--results-file', type=str, default="suffix_-75_30_all_results.pkl",
                        help='Results file to analyze')
    args = parser.parse_args()
    

    output_dir = args.output_dir
    os.makedirs(f"test_grid_search_results/{output_dir}", exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    with open(f"test_grid_search_results/{args.results_file}", 'rb') as f:
        results = pickle.load(f)
    
    # Extract states and rewards
    states = np.array([r[0] for r in results])
    rewards = np.array([r[1] for r in results])
    
    print(f"Loaded {len(states)} states")
    print(f"Reward range: {rewards.min():.4f} to {rewards.max():.4f}")
    print(f"Average reward: {rewards.mean():.4f}")
    
    # Filter for rewards above threshold
    if args.reward_threshold > 0:
        threshold_mask = rewards > args.reward_threshold
        states = states[threshold_mask]
        rewards = rewards[threshold_mask]
        print(f"Filtered to {len(states)} states with rewards > {args.reward_threshold}")
    
    # Apply UMAP for dimensionality reduction
    print("Running UMAP dimensionality reduction...")
    umap_start_time = time.time()
    reducer = umap.UMAP(n_neighbors=15, 
                       min_dist=0.1,
                       n_components=2, 
                       metric='euclidean',
                       random_state=42)
    
    embedding = reducer.fit_transform(states)
    umap_end_time = time.time()
    umap_duration = umap_end_time - umap_start_time
    print(f"UMAP completed in {umap_duration:.2f} seconds ({umap_duration/60:.2f} minutes)")
    
    # Create scatter plot
    print("Creating visualization...")
    plt.figure(figsize=(12, 10))
    
    # Create a color map based on reward values
    norm = Normalize(vmin=rewards.min(), vmax=rewards.max())
    cmap = cm.viridis_r  # Using reversed colormap to highlight high reward states
    
    # Scatter plot with reward-based coloring
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], 
                    c=rewards, 
                    cmap=cmap,
                    alpha=0.7,
                    s=5)  # smaller point size for better visibility
    
    # Add color bar
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), 
                        label='Reward Value')
    
    # Highlight top performers
    top_indices = np.argsort(rewards)[-150:]  # Get indices of top 150
    plt.scatter(embedding[top_indices, 0], embedding[top_indices, 1], 
               c='red', 
               s=20,
               alpha=0.7,
               label='Top 150')
    
    # Add state labels for top 150
    for i, idx in enumerate(top_indices):
        state_str = str(states[idx].tolist())
        reward_str = f" (R: {rewards[idx]:.2f})"
        plt.annotate(state_str + reward_str,
                    (embedding[idx, 0], embedding[idx, 1]),
                    fontsize=2,  # Reduced from 6 to 2
                    alpha=0.5,   # Reduced from 0.6 to 0.5
                    xytext=(5, 5),
                    textcoords='offset points')
    
    title_suffix = f" (Rewards > {args.reward_threshold})" if args.reward_threshold > 0 else ""
    plt.title(f'UMAP Projection of States Colored by Reward{title_suffix}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    threshold_prefix = f"threshold_{args.reward_threshold}_" if args.reward_threshold > 0 else ""
    plt.savefig(f"test_grid_search_results/{output_dir}/{threshold_prefix}reward_umap_visualization.png", dpi=300)
    plt.close()
    
    
    
    
    
    
    
    
    # Create a heat map of the top 2 UMAP dimensions
    print("Creating reward heat map...")
    
    # Create a 2D histogram
    H, xedges, yedges = np.histogram2d(
        embedding[:, 0], embedding[:, 1], 
        bins=50,
        weights=rewards
    )
    
    # Normalize by count
    Hcount, _, _ = np.histogram2d(
        embedding[:, 0], embedding[:, 1], 
        bins=[xedges, yedges]
    )
    
    # Avoid division by zero
    Hcount[Hcount == 0] = 1
    
    # Average reward in each bin
    H = H / Hcount
    
    plt.figure(figsize=(12, 10))
    plt.imshow(H.T, origin='lower', 
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
              aspect='auto',
              cmap='viridis_r')  # Using reversed colormap for heatmap too
    plt.colorbar(label='Average Reward')
    plt.title(f'Average Reward Heat Map in UMAP Space{title_suffix}')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.tight_layout()
    plt.savefig(f"test_grid_search_results/{output_dir}/{threshold_prefix}reward_heatmap.png", dpi=300)
    plt.close()
    
    
    
    
    
    
    # Also generate histograms
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, alpha=0.7)
    plt.title(f'Distribution of Rewards{title_suffix}')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"test_grid_search_results/{output_dir}/{threshold_prefix}reward_distribution.png", dpi=300)
    plt.close()
    
    
    
    print(f"Analysis complete! Visualizations saved to test_grid_search_results/{output_dir}/ directory")

if __name__ == "__main__":
    analyze_results()

