#!/usr/bin/env python3

import os
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, explode, split, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def initialize_spark():
    spark = SparkSession.builder \
        .appName("AdvancedSentimentAnalysis") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_data(spark, csv_path):
    df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(csv_path)
    
    df = df.select(
        col("text"),
        col("label").cast(IntegerType()).alias("label")
    )
    
    return df


def preprocess_text(df):
    df = df.withColumn("text",
        regexp_replace(col("text"), r"http\S+|www\S+|https\S+", "")
    )
    
    df = df.withColumn("text",
        regexp_replace(col("text"), r"\S+@\S+", "")
    )
    
    df = df.withColumn("text",
        regexp_replace(col("text"), r"[^a-zA-Z\s]", "")
    )
    
    df = df.withColumn("text",
        regexp_replace(col("text"), r"\s+", " ")
    )
    
    df = df.withColumn("text", lower(col("text")))
    
    df = df.filter(col("text").rlike(r"\w+"))
    
    return df


def build_word2vec_pipeline(use_naive_bayes=True):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    word2vec = Word2Vec(
        vectorSize=100,
        minCount=2,
        inputCol="filtered_words",
        outputCol="features",
        windowSize=5
    )
    
    if use_naive_bayes:
        classifier = NaiveBayes(
            labelCol="label",
            featuresCol="features",
            smoothing=0.1
        )
    else:
        classifier = LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=100,
            regParam=0.01
        )
    
    pipeline = Pipeline(stages=[tokenizer, remover, word2vec, classifier])
    
    return pipeline


def build_tfidf_pipeline(use_naive_bayes=True):
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    
    hashingTF = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=1000
    )
    
    idf = IDF(inputCol="raw_features", outputCol="features")
    
    if use_naive_bayes:
        classifier = NaiveBayes(
            labelCol="label",
            featuresCol="features",
            smoothing=0.1
        )
    else:
        classifier = LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=100,
            regParam=0.01
        )
    
    pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, classifier])
    
    return pipeline


def run_advanced_analysis(use_word2vec=True, use_naive_bayes=True):
    
    spark = initialize_spark()
    
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'data', 'lab5', 'sent_all_cleaned.csv'
    ))
    
    results_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '..', '..', 'results', 'lab5'
    ))
    os.makedirs(results_dir, exist_ok=True)
    
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("LAB 5 - TASK 4: ADVANCED SENTIMENT ANALYSIS WITH PYSPARK")
    output_lines.append("=" * 80)
    output_lines.append("")
    
    feature_method = "Word2Vec (Spark ML)" if use_word2vec else "TF-IDF (Spark ML)"
    output_lines.append(f"Feature Extraction Method: {feature_method}")
    
    classifier_name = "Naive Bayes" if use_naive_bayes else "Logistic Regression"
    output_lines.append(f"Classifier: {classifier_name}")
    output_lines.append("")
    
    print("[1/6] Loading data...")
    output_lines.append("[1/6] Loading data...")
    try:
        df = load_data(spark, data_path)
        total_count = df.count()
        output_lines.append(f"Total samples: {total_count}")
        
        class_dist = df.groupBy("label").count().collect()
        for row in class_dist:
            output_lines.append(f"  Class {row['label']}: {row['count']} samples")
        output_lines.append("")
        
        print(f"  Loaded {total_count} samples")
    except Exception as e:
        output_lines.append(f"Error loading data: {str(e)}")
        print(f"Error: {str(e)}")
        return
    
    print("[2/6] Advanced preprocessing...")
    output_lines.append("[2/6] Advanced preprocessing...")
    preprocess_start = time.time()
    
    df_processed = preprocess_text(df)
    processed_count = df_processed.count()
    output_lines.append(f"Samples after preprocessing: {processed_count}")
    
    preprocess_time = time.time() - preprocess_start
    output_lines.append(f"Preprocessing time: {preprocess_time:.4f}s")
    output_lines.append("")
    
    print("[3/6] Splitting data (80/20 train/test)...")
    output_lines.append("[3/6] Splitting data (80/20 train/test)...")
    
    train_data, test_data = df_processed.randomSplit([0.8, 0.2], seed=42)
    train_count = train_data.count()
    test_count = test_data.count()
    
    output_lines.append(f"Train set: {train_count} samples")
    output_lines.append(f"Test set: {test_count} samples")
    output_lines.append("")
    
    print("[4/6] Building pipeline...")
    output_lines.append("[4/6] Building pipeline...")
    
    if use_word2vec:
        output_lines.append("Using Word2Vec embeddings (vector_size=100, window=5, minCount=2)")
        pipeline = build_word2vec_pipeline(use_naive_bayes)
    else:
        output_lines.append("Using TF-IDF features (numFeatures=1000)")
        pipeline = build_tfidf_pipeline(use_naive_bayes)
    
    output_lines.append("")
    
    print("[5/6] Training model...")
    output_lines.append("[5/6] Training model...")
    train_start = time.time()
    
    try:
        model = pipeline.fit(train_data)
        train_time = time.time() - train_start
        output_lines.append(f"Training time: {train_time:.4f}s")
        output_lines.append("")
    except Exception as e:
        output_lines.append(f"Error during training: {str(e)}")
        print(f"Error: {str(e)}")
        spark.stop()
        return
    
    print("[6/6] Evaluating model...")
    output_lines.append("[6/6] Evaluating model...")
    eval_start = time.time()
    
    try:
        predictions = model.transform(test_data)
        
        accuracy_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="accuracy"
        )
        accuracy = accuracy_evaluator.evaluate(predictions)
        
        precision_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedPrecision"
        )
        precision = precision_evaluator.evaluate(predictions)
        
        recall_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="weightedRecall"
        )
        recall = recall_evaluator.evaluate(predictions)
        
        f1_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="f1"
        )
        f1_score = f1_evaluator.evaluate(predictions)
        
        eval_time = time.time() - eval_start
        output_lines.append(f"Evaluation time: {eval_time:.4f}s")
        output_lines.append("")
        
        # Compute confusion matrix
        predictions_collect = predictions.select("label", "prediction").collect()
        conf_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        for row in predictions_collect:
            true_label = int(row['label'])
            pred_label = int(row['prediction'])
            conf_matrix[true_label][pred_label] += 1
        
        # Results summary
        output_lines.append("=" * 80)
        output_lines.append("EVALUATION RESULTS")
        output_lines.append("=" * 80)
        output_lines.append(f"Accuracy:  {accuracy:.4f}")
        output_lines.append(f"Precision: {precision:.4f}")
        output_lines.append(f"Recall:    {recall:.4f}")
        output_lines.append(f"F1 Score:  {f1_score:.4f}")
        output_lines.append("")
        
        output_lines.append("Confusion Matrix:")
        output_lines.append("               Predicted")
        output_lines.append("             Class 0  Class 1  Class 2")
        for i, row in enumerate(conf_matrix):
            output_lines.append(f"Actual Class {i}: {row[0]:>6}  {row[1]:>6}  {row[2]:>6}")
        output_lines.append("")
        
        total_time = preprocess_time + train_time + eval_time
        output_lines.append("=" * 80)
        output_lines.append("PERFORMANCE SUMMARY")
        output_lines.append("=" * 80)
        output_lines.append(f"Total processing time: {total_time:.4f}s")
        output_lines.append(f"  - Preprocessing: {preprocess_time:.4f}s")
        output_lines.append(f"  - Training: {train_time:.4f}s")
        output_lines.append(f"  - Evaluation: {eval_time:.4f}s")
        output_lines.append("")
        
    except Exception as e:
        output_lines.append(f"Error during evaluation: {str(e)}")
        print(f"Error: {str(e)}")
    
    # Save results
    output_text = '\n'.join(output_lines)
    
    output_path = os.path.join(results_dir, 'task4.txt')
    
    # Append results to task4.txt (create if not exists)
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(output_text)
        f.write("\n\n")
    
    print("\n" + output_text)
    print(f"\nResults appended to: {output_path}")
    
    spark.stop()
    
    return {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RUNNING ADVANCED SENTIMENT ANALYSIS WITH PYSPARK")
    print("Configuration 1: Word2Vec + Naive Bayes")
    print("=" * 80 + "\n")
    
    results_1 = run_advanced_analysis(use_word2vec=True, use_naive_bayes=True)
    
    print("\n" + "=" * 80)
    print("RUNNING ADVANCED SENTIMENT ANALYSIS WITH PYSPARK")
    print("Configuration 2: Word2Vec + Logistic Regression")
    print("=" * 80 + "\n")
    
    results_2 = run_advanced_analysis(use_word2vec=True, use_naive_bayes=False)
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    if results_1 and results_2:
        print(f"Word2Vec + Naive Bayes     - Accuracy: {results_1['accuracy']:.4f}, F1: {results_1['f1_score']:.4f}")
        print(f"Word2Vec + LogReg          - Accuracy: {results_2['accuracy']:.4f}, F1: {results_2['f1_score']:.4f}")
    print("=" * 80 + "\n")
