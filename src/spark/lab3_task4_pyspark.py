import argparse
import os
import json
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import Word2Vec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default='models/word2vec_spark')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--field', default='text')
    parser.add_argument('--minCount', type=int, default=5)
    parser.add_argument('--vectorSize', type=int, default=100)
    parser.add_argument('--master', default='local[*]')
    args = parser.parse_args()

    print("=" * 60)
    print("PYSPARK WORD2VEC TRAINING")
    print("=" * 60)
    
    spark = SparkSession.builder.master(args.master).appName('lab4_word2vec').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    df = spark.read.json(args.input)
    if args.field not in df.columns:
        print(f"Field '{args.field}' not found in {df.columns}")
        spark.stop()
        return

    df2 = df.select(col(args.field).alias('text')).na.drop()
    df2 = df2.withColumn('clean', lower(col('text')))
    df2 = df2.withColumn('clean', regexp_replace(col('clean'), "[^a-z0-9\\s]", ' '))
    df2 = df2.withColumn('tokens', split(col('clean'), '\\s+'))
    df2 = df2.filter(col('tokens').isNotNull())

    count = df2.count()
    print(f"Dataset: {args.input} ({count} documents)")
    print(f"Vector size: {args.vectorSize}D, Min count: {args.minCount}")
    
    t0 = time.perf_counter()
    w2v = Word2Vec(vectorSize=args.vectorSize, minCount=args.minCount, inputCol='tokens', outputCol='result')
    model = w2v.fit(df2)
    t_train = time.perf_counter() - t0
    
    vocab_size = model.getVectors().count()
    print(f"Vocab size: {vocab_size}")
    print(f"Train time: {t_train:.3f}s")

    test_word = "computer"
    try:
        vec_row = model.getVectors().filter(col('word') == test_word).first()
        if vec_row:
            vec = vec_row['vector'].toArray().tolist()
            print(f"\n[{test_word.upper()} VECTOR ({len(vec)}D)]")
            print(json.dumps(vec))
        
        t0 = time.perf_counter()
        top5 = model.findSynonyms(test_word, 5)
        t_query = time.perf_counter() - t0
        
        print(f"\n[TOP-5 SIMILAR TO '{test_word}']")
        for row in top5.collect():
            print(f"{row['word']}: {row['similarity']:.4f}")
        print(f"Query time: {t_query:.4f}s")
    except Exception as e:
        print(f"Could not find '{test_word}': {e}")

    if not args.no_save:
        model.write().overwrite().save(args.output)
        print(f"\nModel saved to {args.output}")
    
    spark.stop()

if __name__ == '__main__':
    main()

