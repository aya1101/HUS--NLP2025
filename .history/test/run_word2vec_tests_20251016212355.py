#!/usr/bin/env python3
import os
import json
import sys
import time
from pathlib import Path
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'test' / 'data' / 'sample.jsonl'

def load_sample_texts(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                texts.append(json.loads(line).get('text',''))
    return texts

def main():
    print("=" * 60)
    print("GENSIM LOCAL TRAINING TEST")
    print("=" * 60)
    
    texts = load_sample_texts(SAMPLE)
    tokenized = [[w for w in t.lower().split() if w.isalpha()] for t in texts]
    
    t0 = time.perf_counter()
    model = Word2Vec(tokenized, vector_size=50, min_count=1, workers=1, epochs=5)
    t_train = time.perf_counter() - t0
    
    print(f"Dataset: {SAMPLE.name} ({len(texts)} documents)")
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
