import numpy as np
import matplotlib.pyplot as plt
import umap
import pacmap
import phate
import os


def plot_embedding(embedding, color_by, method_name, save_path, cmap, colorbar_label, show_annotations=False, idx=None):
    sort_idx = np.argsort(color_by)  # lowest to highest color value
    embedding_sorted = embedding[sort_idx]
    sorted_color_by = color_by[sort_idx]
    sorted_idx = idx[sort_idx] if idx is not None else None
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embedding_sorted[:, 0], embedding_sorted[:, 1],
                         c=sorted_color_by, cmap=cmap, s=5, alpha=1.0)
    plt.colorbar(scatter, label=colorbar_label)
    
    if show_annotations:
        for i in range(0, len(embedding_sorted), 5):  # Only annotate every 5th index
            annotation = f"  {sorted_idx[i]}" if sorted_idx is not None else f"({embedding_sorted[i, 0]:.2f}, {embedding_sorted[i, 1]:.2f})"
            plt.annotate(annotation, (embedding_sorted[i, 0], embedding_sorted[i, 1]), 
                        fontsize=6, alpha=0.7)
    
    plt.title(f'{method_name} Visualization')
    plt.xlabel(f'{method_name} 1')
    plt.ylabel(f'{method_name} 2')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"visualization_{method_name.lower()}.png"))
    plt.close()



def generate_visualizations(coords, color_by, save_path, show_annotations=False, cmap='viridis_r', colorbar_label='Reward'):
    idx = np.arange(len(coords))
    embeddings = {}
    
    # UMAP
    reducer_umap = umap.UMAP(n_components=2, n_jobs=-1)
    coordinates_umap = reducer_umap.fit_transform(coords)
    plot_embedding(coordinates_umap, color_by, 'UMAP', save_path, cmap, colorbar_label, 
                  show_annotations=show_annotations, idx=idx)
    embeddings['umap'] = {'embedding': coordinates_umap, 'original': coords, 'idx': idx}

    # PaCMAP
    reducer_pacmap = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    coordinates_pacmap = reducer_pacmap.fit_transform(coords)
    plot_embedding(coordinates_pacmap, color_by, 'PaCMAP', save_path, cmap, colorbar_label,
                  show_annotations=show_annotations, idx=idx)
    embeddings['pacmap'] = {'embedding': coordinates_pacmap, 'original': coords, 'idx': idx}

    # PHATE
    reducer_phate = phate.PHATE(knn=300, n_jobs=-1, verbose=True, n_components=2)
    coordinates_phate = reducer_phate.fit_transform(coords)
    plot_embedding(coordinates_phate, color_by, 'PHATE', save_path, cmap, colorbar_label,
                  show_annotations=show_annotations, idx=idx)
    embeddings['phate'] = {'embedding': coordinates_phate, 'original': coords, 'idx': idx}
    
    return embeddings


