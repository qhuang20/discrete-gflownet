import numpy as np
import matplotlib.pyplot as plt
import umap
import pacmap
import phate
import os

def plot_embedding(embedding, color_by, method_name, save_path, cmap, colorbar_label):
    """Helper function to create and save dimensionality reduction plots
    
    Args:
        embedding: The reduced dimensionality coordinates (n_samples, 2)
        color_by: Array of values to color the points by
        method_name: Name of the dimensionality reduction method
        save_path: Path to save the visualization plot
        cmap: Colormap to use for the scatter plot (default: 'viridis_r')
        colorbar_label: Label for the colorbar
    """
    # Sort points by color values so higher values are plotted last (on top)
    sort_idx = np.argsort(color_by) # lowest to highest 
    embedding_sorted = embedding[sort_idx]
    sorted_color_by = color_by[sort_idx]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_sorted[:, 0], embedding_sorted[:, 1],
                         c=sorted_color_by, cmap=cmap, s=100, alpha=0.6)
    plt.colorbar(scatter, label=colorbar_label)
    plt.title(f'{method_name} Visualization')
    plt.xlabel(f'{method_name} 1')
    plt.ylabel(f'{method_name} 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"visualization_{method_name.lower()}.png"))
    plt.close()

def generate_visualizations(coords, color_by, save_path, cmap='viridis_r', colorbar_label='Reward'):
    """Generate dimensionality reduction visualizations using multiple methods
    
    Args:
        coords: Array of state vectors to visualize
        color_by: Array of values to color the points by (e.g. rewards)
        save_path: Path to save the visualization plots
        cmap: Colormap to use for the scatter plots (default: 'viridis_r')
        colorbar_label: Label for the colorbar (default: 'Reward')
    """
    # UMAP visualization
    reducer_umap = umap.UMAP(n_components=2, n_jobs=-1)
    coordinates_umap = reducer_umap.fit_transform(coords)
    plot_embedding(coordinates_umap, color_by, 'UMAP', save_path, cmap, colorbar_label)

    # PaCMAP visualization  
    reducer_pacmap = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    coordinates_pacmap = reducer_pacmap.fit_transform(coords)
    plot_embedding(coordinates_pacmap, color_by, 'PaCMAP', save_path, cmap, colorbar_label)

    # PHATE visualization
    reducer_phate = phate.PHATE(knn=300, n_jobs=-1, verbose=True, n_components=2)
    coordinates_phate = reducer_phate.fit_transform(coords)
    plot_embedding(coordinates_phate, color_by, 'PHATE', save_path, cmap, colorbar_label)




