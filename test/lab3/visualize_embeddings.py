#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if ROOT not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.representations.word_embedder import WordEmbedder

def visualize_embeddings(model_name="glove-wiki-gigaword-50", n_words=100):
    print("=" * 60)
    print("WORD EMBEDDING VISUALIZATION (PCA)")
    print("=" * 60)
    
    we = WordEmbedder(model_name)
    print(f"Model: {model_name}")
    
    vocab = list(we.model.key_to_index.keys())[:n_words]
    vectors = np.array([we.get_vector(word) for word in vocab])
    
    print(f"Visualizing {len(vocab)} words")
    print(f"Original dimension: {vectors.shape[1]}D")
    
    pca_2d = PCA(n_components=2)
    vectors_2d = pca_2d.fit_transform(vectors)
    
    pca_3d = PCA(n_components=3)
    vectors_3d = pca_3d.fit_transform(vectors)
    
    print(f"Reduced to 2D using PCA")
    print(f"Explained variance (2D): {pca_2d.explained_variance_ratio_.sum():.2%}")
    print(f"Explained variance (3D): {pca_3d.explained_variance_ratio_.sum():.2%}")
    
    fig = plt.figure(figsize=(20, 8))
    
    ax1 = fig.add_subplot(121)
    ax1.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5, s=30, c='steelblue')
    
    for i, word in enumerate(vocab):
        ax1.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    ax1.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax1.set_title(f'2D Visualization (PCA)\n{model_name} - Top {n_words} words')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2], 
               alpha=0.5, s=30, c='coral')
    
    for i, word in enumerate(vocab):
        ax2.text(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2], 
                word, fontsize=7, alpha=0.7)
    
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
    ax2.set_title(f'3D Visualization (PCA)\n{model_name} - Top {n_words} words')
    
    plt.tight_layout()
    
    output_path = ROOT / 'results' / 'embedding_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    return output_path

if __name__ == '__main__':
    visualize_embeddings()
