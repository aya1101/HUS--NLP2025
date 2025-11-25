import argparse
import re
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace, split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='test/data/sample.jsonl', help='Path to JSON/JSONL input (default: test/data/sample.jsonl)')
    parser.add_argument('--field', default='text', help='JSON text field name')
    parser.add_argument('--vectorSize', type=int, default=100)
    parser.add_argument('--minCount', type=int, default=5)
    args = parser.parse_args()

    spark = SparkSession.builder.master('local[*]').appName('lab4_spark_word2vec_demo').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    print('Reading input:', args.input)
    df = spark.read.json(args.input)

    if args.field not in df.columns:
        print(f"Field '{args.field}' not found. Columns: {df.columns}")
        spark.stop()
        return

    # Preprocessing: select text, lowercase, remove non-word characters, split
    df2 = df.select(col(args.field).alias('text')).na.drop()
    df2 = df2.withColumn('clean', lower(col('text')))
    df2 = df2.withColumn('clean', regexp_replace(col('clean'), "[^a-z0-9\\s]", ' '))
    df2 = df2.withColumn('tokens', split(col('clean'), '\\s+'))

    # Filter empty tokens
    df2 = df2.filter((col('tokens').isNotNull()))

    print('Count after preprocessing:', df2.count())

    # Train Word2Vec
    w2v = Word2Vec(vectorSize=args.vectorSize, minCount=args.minCount, inputCol='tokens', outputCol='result')
    model = w2v.fit(df2)

    # Find synonyms for 'computer'
    try:
        words = model.findSynonyms('computer', 5)
        print('Top-5 similar to computer:')
        words.show()
    except Exception as e:
        print('Could not find synonyms for "computer":', e)

    spark.stop()


if __name__ == '__main__':
    main()
