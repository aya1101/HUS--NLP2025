#!/usr/bin/env python3
import sys
import os
from typing import List, Union
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_dataset, DatasetDict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectoriser import CountVectorizer
from src.model.text_classifier import TextClassifier
import pandas as pd

texts = [
	"This movie is fantastic and I love it!",
	"I hate this film, it's terrible.",
	"The acting was superb, a truly great experience.",
	"What a waste of time, absolutely boring.",
	"Highly recommend this, a masterpiece.",
	"Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Instantiate tokenizer and vectorizer
tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer=tokenizer)

# Instantiate classifier
clf = TextClassifier(vectorizer)


# Measure training and prediction time
start_time = time.time()
clf.fit(X_train, y_train)
train_time = time.time() - start_time

start_pred_time = time.time()
y_pred = clf.predict(X_test)
pred_time = time.time() - start_pred_time

# Evaluate
metrics = clf.evaluate(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Prepare output
output_lines = []
output_lines.append(f"Model training time: {train_time:.4f} seconds")
output_lines.append(f"Model prediction time: {pred_time:.4f} seconds")
output_lines.append("Evaluation metrics:")
for k, v in metrics.items():
	output_lines.append(f"{k}: {v:.3f}")

output_lines.append("\nConfusion Matrix:")
output_lines.append(str(cm))

# Save to results folder
results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'lab5')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'task1_2_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
	f.write('\n'.join(output_lines))

# Also print to console
for line in output_lines:
	print(line)


# === Additional test: train on sent_train.csv, evaluate on sent_valid.csv ===
print('\n=== Running additional test: train on data/lab5/sent_train.csv and evaluate on sent_valid.csv ===')

train_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lab5', 'sent_train.csv')
valid_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lab5', 'sent_valid.csv')

df_train = pd.read_csv(train_path)
df_valid = pd.read_csv(valid_path)

# Ensure no nulls and cast types
df_train = df_train.dropna(subset=['text', 'label'])
df_valid = df_valid.dropna(subset=['text', 'label'])

texts_train = df_train['text'].astype(str).tolist()
labels_train = df_train['label'].astype(int).tolist()

texts_valid = df_valid['text'].astype(str).tolist()
labels_valid = df_valid['label'].astype(int).tolist()

# New tokenizer/vectorizer/classifier instances
tokenizer2 = RegexTokenizer()
vectorizer2 = CountVectorizer(tokenizer=tokenizer2)
clf2 = TextClassifier(vectorizer2)

# Train on sent_train.csv
start_time2 = time.time()
clf2.fit(texts_train, labels_train)
train_time2 = time.time() - start_time2

# Predict on sent_valid.csv
start_pred2 = time.time()
y_pred2 = clf2.predict(texts_valid)
pred_time2 = time.time() - start_pred2

# Evaluate
metrics2 = clf2.evaluate(labels_valid, y_pred2)
cm2 = confusion_matrix(labels_valid, y_pred2)

# Prepare output
out2 = []
out2.append('ADDITIONAL TEST: train on sent_train.csv, evaluate on sent_valid.csv')
out2.append(f"Train samples: {len(texts_train)}; Validation samples: {len(texts_valid)}")
out2.append(f"Model training time: {train_time2:.4f} seconds")
out2.append(f"Model prediction time: {pred_time2:.4f} seconds")
out2.append('Evaluation metrics:')
for k, v in metrics2.items():
	out2.append(f"{k}: {v:.3f}")
out2.append('\nConfusion Matrix:')
out2.append(str(cm2))

# Save to results folder
results_dir2 = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'lab5')
os.makedirs(results_dir2, exist_ok=True)
output_path2 = os.path.join(results_dir2, 'task2_train_valid_results.txt')
with open(output_path2, 'w', encoding='utf-8') as f:
	f.write('\n'.join(out2))

for line in out2:
	print(line)

print(f"\nAdditional test results saved to: {output_path2}")