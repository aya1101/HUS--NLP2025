#!/usr/bin/env python3
import sys
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / 'results' / 'log4.txt'

class Tee:
    def __init__(self, filepath):
        self.filepath = filepath
        self._file = None
        self._stdout = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self._file = open(self.filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, data):
        if self._stdout:
            self._stdout.write(data)
        if self._file:
            self._file.write(data)
            if "\n" in data:
                try:
                    self._file.flush()
                except:
                    pass

    def flush(self):
        if self._stdout:
            self._stdout.flush()
        if self._file:
            self._file.flush()

    def __exit__(self, exc_type, exc, tb):
        sys.stdout = self._stdout
        if self._file:
            self._file.close()

def run_test(script_path, args=[]):
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr and 'WARN' not in result.stderr:
        print(result.stderr)
    return result.returncode

def main():
    with Tee(LOG_FILE):
        print("\n" + "=" * 60)
        print("TEST 1: GENSIM PRETRAINED MODEL")
        print("=" * 60 + "\n")
        run_test(ROOT / 'test' / 'lab4_test.py')
        
        print("\n\n" + "=" * 60)
        print("TEST 2: GENSIM LOCAL TRAINING")
        print("=" * 60 + "\n")
        run_test(ROOT / 'test' / 'run_word2vec_tests.py')
        
        print("\n\n" + "=" * 60)
        print("TEST 3: PYSPARK TRAINING")
        print("=" * 60 + "\n")
        run_test(
            ROOT / 'src' / 'spark' / 'lab4_task4_pyspark.py',
            ['--input', 'test/data/sample.jsonl', '--field', 'text', 
             '--vectorSize', '50', '--minCount', '1', '--no-save']
        )
        
        print("\n" + "=" * 60)
        print(f"ALL TESTS COMPLETED - Results saved to {LOG_FILE}")
        print("=" * 60)

if __name__ == '__main__':
    main()
