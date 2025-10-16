#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import json
import time

ROOT = Path(__file__).resolve().parents[1]
if ROOT not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_c4_texts(path, max_docs=30000):
    texts = []
    with open(path, 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if i >= max_docs:
                break
            if line.strip():
                texts.append(json.loads(line).get('text', ''))
    return texts

def train_and_visualize():
    print("=" * 60)
    print("WORD EMBEDDING VISUALIZATION FROM TRAINED MODEL")
    print("=" * 60)
    
    c4_path = ROOT / 'data' / 'c4-train.00000-of-01024-30K.json'
    
    print("Loading C4 dataset...")
    texts = load_c4_texts(c4_path)
    tokenized = [[w for w in t.lower().split() if w.isalpha()] for t in texts]
    
    print(f"Training Word2Vec on {len(texts)} documents...")
    t0 = time.perf_counter()
    model = Word2Vec(tokenized, vector_size=100, min_count=5, workers=4, epochs=3)
    t_train = time.perf_counter() - t0
    print(f"Train time: {t_train:.1f}s")
    print(f"Vocab size: {len(model.wv.key_to_index)}")
    
    selected_words = []
    categories = {
        'Technology': ['computer', 'software', 'technology', 'internet', 'digital', 'data'],
        'Science': ['research', 'study', 'science', 'university', 'theory', 'experiment'],
        'Business': ['company', 'business', 'market', 'industry', 'economy', 'trade'],
        'Politics': ['government', 'president', 'political', 'election', 'policy', 'congress'],
        'Common': ['people', 'time', 'world', 'year', 'way', 'work']
    }
    
    word_labels = []
    word_colors = []
    color_map = {
        'Technology': 'red',
        'Science': 'blue', 
        'Business': 'green',
        'Politics': 'orange',
        'Common': 'gray'
    }
    
    for category, words in categories.items():
        for word in words:
            if word in model.wv:
                selected_words.append(word)
                word_labels.append(f"{word} ({category})")
                word_colors.append(color_map[category])
    
    if len(selected_words) < 5:
        print("Not enough words in vocabulary, using top 50 words instead")
        selected_words = list(model.wv.key_to_index.keys())[:50]
        word_colors = ['blue'] * len(selected_words)
        word_labels = selected_words
    
    vectors = np.array([model.wv[word] for word in selected_words])
    
    print(f"\nVisualizing {len(selected_words)} words")
    print(f"Original dimension: {vectors.shape[1]}D")
    
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    print(f"Reduced to 2D using PCA")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    for category, color in color_map.items():
        indices = [i for i, c in enumerate(word_colors) if c == color]
        if indices:
            ax1.scatter(vectors_2d[indices, 0], vectors_2d[indices, 1], 
                       c=color, label=category, alpha=0.6, s=100)
    
    for i, word in enumerate(selected_words):
        ax1.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), 
                    fontsize=9, alpha=0.8)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax1.set_title('Word Embeddings by Category\n(Gensim Word2Vec, C4 Dataset)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    top_100 = list(model.wv.key_to_index.keys())[:100]
    vectors_100 = np.array([model.wv[word] for word in top_100])
    vectors_100_2d = pca.transform(vectors_100)
    
    ax2.scatter(vectors_100_2d[:, 0], vectors_100_2d[:, 1], alpha=0.4, s=30, c='steelblue')
    
    for i in range(0, len(top_100), 5):
        ax2.annotate(top_100[i], xy=(vectors_100_2d[i, 0], vectors_100_2d[i, 1]), 
                    fontsize=7, alpha=0.6)
    
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax2.set_title('Top 100 Most Frequent Words\n(Gensim Word2Vec, C4 Dataset)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = ROOT / 'results' / 'embedding_visualization_trained.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to: {output_path}")
    
    return output_path

if __name__ == '__main__':
    train_and_visualize()
