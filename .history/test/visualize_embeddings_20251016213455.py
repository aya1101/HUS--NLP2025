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
    
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    print(f"Reduced to 2D using PCA")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    plt.figure(figsize=(14, 10))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.5, s=30)
    
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                    fontsize=8, alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    plt.title(f'Word Embeddings Visualization - {model_name}\n(Top {n_words} words, PCA 2D)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = ROOT / 'results' / 'embedding_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    return output_path

if __name__ == '__main__':
    visualize_embeddings()
