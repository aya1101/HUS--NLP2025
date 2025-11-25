#!/usr/bin/env python3
import sys
import os
import json
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.representations.word_embedder import WordEmbedder

def main():
    model_name = "glove-wiki-gigaword-50"
    print("=" * 60)
    print("GENSIM PRETRAINED MODEL TEST")
    print("=" * 60)
    
    t0 = time.perf_counter()
    we = WordEmbedder(model_name)
    t_load = time.perf_counter() - t0
    print(f"Model: {model_name}")
    print(f"Load time: {t_load:.3f}s")
    
    test_word = "computer"
    vec = we.get_vector(test_word)
    print(f"\n[{test_word.upper()} VECTOR ({len(vec)}D)]")
    print(json.dumps(vec.tolist()))
    
    t0 = time.perf_counter()
    top5 = we.model.most_similar(test_word, topn=5)
    t_query = time.perf_counter() - t0
    
    print(f"\n[TOP-5 SIMILAR TO '{test_word}']")
    for word, score in top5:
        print(f"{word}: {score:.4f}")
    print(f"Query time: {t_query:.4f}s")

if __name__ == '__main__':
    main()
