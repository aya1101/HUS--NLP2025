"""
Lab 3 - NLP Pipeline with PySpark
Nguyễn Thùy Trang - 22000128
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import explode, length, col
import time
import os

# Configuration variables
LIMIT_DOCUMENTS = 1000  # Easily customizable document limit
DATA_PATH = "data/c4-train.00000-of-01024-30K.json.gz"
NUM_FEATURES = 20000

def create_spark_session():
    """Khởi tạo và cấu hình SparkSession"""
    spark = SparkSession.builder \
        .appName("Lab3 NLP Pipeline") \
        .master("local[*]") \
        .getOrCreate()
    
    print("Spark Session created successfully.")
    print("Spark UI available at http://localhost:4040")
    print("Pausing for 10 seconds to allow you to open the Spark UI...")
    time.sleep(10)
    
    return spark

def load_data(spark, data_path=DATA_PATH, limit=LIMIT_DOCUMENTS):
    """Đọc và load dữ liệu từ file JSON"""
    df = spark.read.json(data_path).limit(limit)
    
    print(f"Successfully loaded {df.count()} records (limited to {limit}).")
    df.printSchema()
    print("\nSample of initial DataFrame:")
    df.show(5, truncate=False)
    
    return df

def create_pipeline():
    """Tạo NLP pipeline với các stages"""
    # Tokenization
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="\\s+|[.,;!?()\"']"
    )
    
    # Stop Words Removal
    stop_words_remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )
    
    # Vectorization (Term Frequency)
    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        numFeatures=NUM_FEATURES
    )
    
    # Vectorization (Inverse Document Frequency)
    idf = IDF(
        inputCol="raw_features",
        outputCol="features"
    )
    
    # Tạo Pipeline
    pipeline = Pipeline(stages=[tokenizer, stop_words_remover, hashing_tf, idf])
    
    return pipeline

def fit_and_transform_pipeline(pipeline, df):
    """Fit pipeline và transform dữ liệu"""
    print("\nFitting the NLP pipeline...")
    fit_start_time = time.time()
    pipeline_model = pipeline.fit(df)
    fit_duration = time.time() - fit_start_time
    print(f"--> Pipeline fitting took {fit_duration:.2f} seconds.")
    
    print("\nTransforming data with the fitted pipeline...")
    transform_start_time = time.time()
    transformed_df = pipeline_model.transform(df)
    transformed_df.cache()
    transform_count = transformed_df.count()
    transform_duration = time.time() - transform_start_time
    print(f"--> Data transformation of {transform_count} records took {transform_duration:.2f} seconds.")
    
    return transformed_df, fit_duration, transform_duration

def analyze_vocabulary(transformed_df):
    """Phân tích kích thước từ vựng"""
    actual_vocab_size = transformed_df \
        .select(explode(col("filtered_tokens")).alias("word")) \
        .filter(length(col("word")) > 1) \
        .distinct() \
        .count()
    print(f"--> Actual vocabulary size after preprocessing: {actual_vocab_size} unique terms.")
    
    return actual_vocab_size

def save_results(transformed_df, fit_duration, transform_duration, actual_vocab_size):
    """Lưu kết quả và metrics"""
    os.makedirs("results", exist_ok=True)
    
    # Lưu metrics
    with open("results/lab3_metrics.log", "w", encoding="utf-8") as f:
        f.write("--- Performance Metrics ---\n")
        f.write(f"Pipeline fitting duration: {fit_duration:.2f} seconds\n")
        f.write(f"Data transformation duration: {transform_duration:.2f} seconds\n")
        f.write(f"Actual vocabulary size (after preprocessing): {actual_vocab_size} unique terms\n")
        f.write(f"HashingTF numFeatures set to: 20000\n")
        if 20000 < actual_vocab_size:
            f.write(f"Note: numFeatures (20000) is smaller than actual vocabulary size ({actual_vocab_size}). Hash collisions are expected.\n")
    
    print("Successfully wrote metrics to results/lab3_metrics.log")
    
    # Lưu kết quả pipeline
    results = transformed_df.select("text", "features").take(20)
    with open("results/lab3_pipeline_output.txt", "w", encoding="utf-8") as f:
        f.write("--- NLP Pipeline Output (First 20 results) ---\n\n")
        for i, row in enumerate(results):
            text = row["text"]
            features = row["features"]
            f.write("=" * 80 + "\n")
            f.write(f"Record {i+1}:\n")
            f.write(f"Original Text: {text[:100]}...\n")
            f.write(f"TF-IDF Vector: {str(features)}\n")
            f.write("=" * 80 + "\n\n")
    
    print("Successfully wrote 20 results to results/lab3_pipeline_output.txt")

def main():
    """Main function - orchestrate toàn bộ pipeline"""
    # 1. Khởi tạo SparkSession
    spark = create_spark_session()
    
    try:
        # 2. Load dữ liệu
        df = load_data(spark)
        
        # 3. Tạo pipeline
        pipeline = create_pipeline()
        
        # 4. Fit và transform
        transformed_df, fit_duration, transform_duration = fit_and_transform_pipeline(pipeline, df)
        
        # 5. Phân tích từ vựng
        actual_vocab_size = analyze_vocabulary(transformed_df)
        
        # 6. Hiển thị kết quả mẫu
        print("\nSample of transformed data:")
        transformed_df.select("text", "features").show(5, truncate=False)
        
        # 7. Lưu kết quả
        save_results(transformed_df, fit_duration, transform_duration, actual_vocab_size)
        
    finally:
        spark.stop()
        print("Spark Session stopped.")

if __name__ == "__main__":
    main()