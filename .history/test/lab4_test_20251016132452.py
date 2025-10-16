#!/usr/bin/env python3
"""Test script for Lab4: WordEmbedder + DocumentEmbedder

This script will:
 - instantiate WordEmbedder with a gensim model name
 - retrieve the vector for 'king'
 - compute similarity between 'king' and 'queen', and 'king' and 'man'
 - print 10 most similar words to 'computer'
 - embed the sentence "The queen rules the country." and print the document vector

Run from project root:
    python test/lab4_test.py
"""

import sys
import os

# Ensure project root is on sys.path so `from src...` imports work
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.representations.word_embedder import WordEmbedder
from src.representations.document_embedder import DocumentEmbedder
import numpy as np


def pretty(arr, n=8):
    arr = np.asarray(arr)
    return "[" + ", ".join(f"{float(x):.6f}" for x in arr[:n]) + (", ...]" if arr.size > n else "]")


def main():
    model_name = "glove-wiki-gigaword-50"
    print(f"Loading embedding model: {model_name}")

    try:
        we = WordEmbedder(model_name)
    except Exception as e:
        print(f"Không thể tải model {model_name}: {e}")
        return

    print("Model loaded.")

    # 1) Vector for 'king'
    try:
        king_vec = we.get_vector("king")
        print("\nVector cho 'king' (first dims):", pretty(king_vec, 8))
        print("Kích thước vector:", len(king_vec))
    except KeyError:
        print("Từ 'king' không có trong vốn từ của mô hình.")

    # 2) Similarities
    try:
        sim_k_q = we.model.similarity("king", "queen")
        sim_k_m = we.model.similarity("king", "man")
        print(f"\nSimilarity('king','queen') = {sim_k_q:.4f}")
        print(f"Similarity('king','man') = {sim_k_m:.4f}")
    except Exception as e:
        # fallback: use cosine similarity via vectors
        print("Không thể gọi model.similarity(), thử tính bằng vector nếu có...")
        try:
            qv = we.get_vector("queen")
            mv = we.get_vector("man")
            import numpy as _np
            def cos(a,b):
                a = _np.asarray(a, dtype=float)
                b = _np.asarray(b, dtype=float)
                return float(_np.dot(a,b) / (_np.linalg.norm(a)*_np.linalg.norm(b)))
            print(f"Similarity('king','queen') [cosine] = {cos(king_vec,qv):.4f}")
            print(f"Similarity('king','man') [cosine] = {cos(king_vec,mv):.4f}")
        except Exception as e2:
            print("Không thể tính similarity bằng vector:", e2)

    # 3) Most similar to 'computer'
    try:
        print("\nTop 10 từ tương tự với 'computer':")
        for word, score in we.model.most_similar("computer", topn=10):
            print(f"  {word}: {score:.4f}")
    except Exception as e:
        print("Lỗi khi gọi most_similar:", e)

    # 4) Embed a sentence
    de = DocumentEmbedder(we)
    sentence = "The queen rules the country."
    print(f"\nEmbedding câu: \"{sentence}\"")
    try:
        doc_vec = de.embed(sentence)
        print("Document vector (first dims):", pretty(doc_vec, 8))
        print("Document vector shape:", doc_vec.shape)
    except Exception as e:
        print("Lỗi khi embed câu:", e)


if __name__ == '__main__':
    main()
