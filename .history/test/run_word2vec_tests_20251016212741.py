#!/usr/bin/env python3
import os
import json
import sys
import time
from pathlib import Path
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
C4_DATASET = ROOT / 'data' / 'c4-train.00000-of-01024-30K.json'

def load_texts(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                texts.append(item.get('text', ''))
        else:
            texts.append(data.get('text', ''))
    return texts

def main():
    print("=" * 60)
    print("GENSIM LOCAL TRAINING TEST")
    print("=" * 60)
    
    texts = load_texts(C4_DATASET)
    tokenized = [[w for w in t.lower().split() if w.isalpha()] for t in texts]
    
    t0 = time.perf_counter()
    model = Word2Vec(tokenized, vector_size=100, min_count=5, workers=4, epochs=3)
    t_train = time.perf_counter() - t0
    
    print(f"Dataset: {C4_DATASET.name} ({len(texts)} documents)")
    print(f"Vocab size: {len(model.wv.key_to_index)}")
    print(f"Train time: {t_train:.3f}s")
    
    test_word = "computer"
    vec = model.wv[test_word]
    print(f"\n[{test_word.upper()} VECTOR ({len(vec)}D)]")
    print(json.dumps(vec.tolist()))
    
    t0 = time.perf_counter()
    top5 = model.wv.most_similar(test_word, topn=5)
    t_query = time.perf_counter() - t0
    
    print(f"\n[TOP-5 SIMILAR TO '{test_word}']")
    for word, score in top5:
        print(f"{word}: {score:.4f}")
    print(f"Query time: {t_query:.4f}s")

if __name__ == '__main__':
    main()
