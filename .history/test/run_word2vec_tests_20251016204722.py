#!/usr/bin/env python3
"""Run two training tests:
 1) gensim Word2Vec on a small sample (local)
 2) PySpark Word2Vec using src/spark/lab4_task4_pyspark.py on the same sample

Results are written to results/log_train.txt
"""
import os
import json
import subprocess
import sys
from pathlib import Path

import gensim
from gensim.models import Word2Vec

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / 'test' / 'data' / 'sample.jsonl'
LOG = ROOT / 'results' / 'log_train.txt'
SPARK_SCRIPT = ROOT / 'src' / 'spark' / 'lab4_task4_pyspark.py'

os.makedirs(ROOT / 'results', exist_ok=True)

def load_sample_texts(path):
    texts = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(obj.get('text',''))
    return texts

def train_gensim(texts):
    tokenized = [ [w for w in t.lower().split() if w.isalpha()] for t in texts ]
    model = Word2Vec(tokenized, vector_size=50, min_count=1, workers=1, epochs=5)
    return model

def run_spark(sample_path):
    cmd = [sys.executable, str(SPARK_SCRIPT), '--input', str(sample_path), '--field', 'text', '--output', 'models/word2vec_spark_test', '--vectorSize', '50', '--minCount', '1', '--no-save']
    # run and capture stdout/stderr
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc

def main():
    texts = load_sample_texts(SAMPLE)
    with open(LOG, 'w', encoding='utf-8') as fh:
        fh.write('=== GENSIM TRAIN ===\n')
        # train gensim
        gm = train_gensim(texts)
        fh.write('Vocab size (gensim): %d\n' % len(gm.wv.key_to_index))
        fh.write('Most similar to computer: %s\n' % str(gm.wv.most_similar('computer', topn=5)))

        fh.write('\n=== SPARK TRAIN ===\n')
        proc = run_spark(SAMPLE)
        fh.write('returncode: %d\n' % proc.returncode)
        fh.write('stdout:\n')
        fh.write(proc.stdout + '\n')
        fh.write('stderr:\n')
        fh.write(proc.stderr + '\n')

    print('Wrote train log to', LOG)

if __name__ == '__main__':
    main()
