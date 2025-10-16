"""Train Word2Vec on large text datasets using PySpark MLlib.

Usage (PowerShell):
	& ".\ .venv\Scripts\Activate.ps1"; \
	python src/spark/lab4_task4_pyspark.py --input data.jsonl --output models/word2vec_spark --field text --minCount 5 --vectorSize 100

The script:
 - loads JSON/JSONL files where each record has a text field (default: "text")
 - basic preprocessing: lowercase, simple tokenization (RegexTokenizer), remove stopwords
 - trains pyspark.ml.feature.Word2Vec
 - saves the model (Spark's Word2VecModel) and writes a small `results/word2vec_sample.txt` with sample vectors

Note: On Windows you may need to set JAVA_HOME and have a compatible Java runtime. Installing PySpark in the project's venv is recommended:
	pip install pyspark

"""
import argparse
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec


def build_spark(master_url: str = 'local[*]'):
	spark = SparkSession.builder.master(master_url).appName('lab4_word2vec_train').getOrCreate()
	spark.sparkContext.setLogLevel('WARN')
	return spark


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--input', '-i', required=True, help='Input JSON or JSONL file or directory')
	parser.add_argument('--output', '-o', default='models/word2vec_spark', help='Output directory for the saved model')
	parser.add_argument('--no-save', action='store_true', help='Do not call model.save() (useful on Windows without winutils)')
	parser.add_argument('--field', default='text', help='JSON field containing text')
	parser.add_argument('--minCount', type=int, default=5)
	parser.add_argument('--vectorSize', type=int, default=100)
	parser.add_argument('--epochs', type=int, default=1)
	parser.add_argument('--master', default='local[*]', help='Spark master URL')
	args = parser.parse_args()

	os.makedirs('results', exist_ok=True)

	spark = build_spark(args.master)

	print('Reading input:', args.input)
	df = spark.read.option('multiline', 'false').json(args.input)

	if args.field not in df.columns:
		print(f"Field '{args.field}' not found in input columns: {df.columns}")
		spark.stop()
		raise SystemExit(1)

	# Tokenize and remove stopwords using Spark transformers
	tokenizer = RegexTokenizer(inputCol=args.field, outputCol='tokens_raw', pattern='\\W+')
	remover = StopWordsRemover(inputCol='tokens_raw', outputCol='tokens')

	df2 = tokenizer.transform(df.select(col(args.field))).select('tokens_raw')
	df2 = remover.transform(df2).select('tokens')

	# Drop empty token lists
	df2 = df2.filter(col('tokens').isNotNull())

	cnt = df2.count()
	print('Records after tokenization:', cnt)

	if cnt == 0:
		print('No data to train on. Exiting.')
		spark.stop()
		return

	print('Training Word2Vec...')
	w2v = Word2Vec(vectorSize=args.vectorSize, minCount=args.minCount, inputCol='tokens', outputCol='result')
	model = w2v.fit(df2)

	if args.no_save:
		print('Skipping model.save() because --no-save was set')
	else:
		print('Saving model to', args.output)
		model.write().overwrite().save(args.output)

	# write sample vectors (first 200) to results/word2vec_sample.txt
	try:
		vectors = model.getVectors().take(200)
		sample_file = os.path.join('results', 'word2vec_sample.txt')
		with open(sample_file, 'w', encoding='utf-8') as fh:
			for row in vectors:
				fh.write(f"{row['word']}\t{json.dumps(row['vector'])}\n")
		print('Wrote sample vectors to', sample_file)
	except Exception as e:
		print('Could not write sample vectors:', e)

	spark.stop()
	print('Done')


if __name__ == '__main__':
	main()

