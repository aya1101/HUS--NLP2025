"""
Lab 3 - NLP Pipeline with PySpark
Nguyễn Thùy Trang - 22000128
"""

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline,  PipelineModel
from pyspark.sql import functions as F
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.sql.functions import explode, length, col
import time
import os

# Configuration variables
LIMIT_DOCUMENTS = 2000  # Easily customizable document limit
DATA_PATH = "UD_English-EWT/en_ewt-ud-train.txt"  # Updated to use available data
NUM_FEATURES = 20000
TOP_K_SIMILAR = 5  # Number of top similar documents to return

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
    """Đọc và load dữ liệu từ file text"""
    # Read text file as DataFrame
    df = spark.read.text(data_path) \
        .withColumnRenamed("value", "text") \
        .filter(F.length(F.col("text")) > 10) \
        .limit(limit)
    
    print(f"Successfully loaded {df.count()} records (limited to {limit}).")
    df.printSchema()
    print("\nSample of initial DataFrame:")
    df.show(5, truncate=False)
    
    return df

def create_pipeline():
    """Tạo Spark ML Pipeline với tất cả các stage cần thiết"""
    print("\nCreating Spark ML Pipeline with stages:")
    
    # Stage 1: Tokenization using RegexTokenizer
    print("1. RegexTokenizer - Tokenizing text using regex pattern")
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="\\s+|[.,;!?()\"']"
    )
    
    # Stage 2: Stop Words Removal
    print("2. StopWordsRemover - Removing English stop words")
    stop_words_remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )
    
    # Stage 3: Term Frequency using HashingTF
    print(f"3. HashingTF - Computing term frequencies with {NUM_FEATURES} features")
    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        numFeatures=NUM_FEATURES
    )
    
    # Stage 4: Inverse Document Frequency
    print("4. IDF - Computing inverse document frequencies")
    idf = IDF(
        inputCol="raw_features",
        outputCol="tfidf_features"
    )
    
    # Stage 5: Normalization
    print("5. Normalizer - L2 normalization of TF-IDF vectors")
    normalizer = Normalizer(
        inputCol="tfidf_features",
        outputCol="features",
        p=2.0
    )

    # Create Pipeline with all stages
    pipeline = Pipeline(stages=[tokenizer, stop_words_remover, hashing_tf, idf, normalizer])
    print("Pipeline created successfully with 5 stages.\n")

    return pipeline

def fit_and_transform_pipeline(pipeline, df):
    """Fit pipeline và transform dữ liệu với detailed performance measurement"""
    print("\n" + "="*60)
    print("PIPELINE FITTING AND TRANSFORMATION")
    print("="*60)
    
    # Stage 1: Pipeline Fitting
    print("\nStage 1: Fitting the NLP pipeline...")
    fit_start_time = time.time()
    pipeline_model = pipeline.fit(df)
    fit_duration = time.time() - fit_start_time
    print(f"✓ Pipeline fitting completed in {fit_duration:.2f} seconds")
    
    # Stage 2: Data Transformation
    print("\nStage 2: Transforming data with the fitted pipeline...")
    transform_start_time = time.time()
    transformed_df = pipeline_model.transform(df)
    
    # Cache the result for better performance in subsequent operations
    print("Caching transformed DataFrame for better performance...")
    transformed_df.cache()
    
    # Count records to trigger caching
    transform_count = transformed_df.count()
    transform_duration = time.time() - transform_start_time
    print(f"✓ Data transformation of {transform_count:,} records completed in {transform_duration:.2f} seconds")
    
    # Performance summary
    total_pipeline_time = fit_duration + transform_duration
    print(f"\n📊 Pipeline Performance Summary:")
    print(f"   • Fitting time: {fit_duration:.2f}s")
    print(f"   • Transform time: {transform_duration:.2f}s")
    print(f"   • Total pipeline time: {total_pipeline_time:.2f}s")
    print(f"   • Records processed: {transform_count:,}")
    print(f"   • Processing rate: {transform_count/total_pipeline_time:.1f} records/second")
    
    return transformed_df, pipeline_model, fit_duration, transform_duration

def analyze_vocabulary(transformed_df):
    """Phân tích kích thước từ vựng"""
    actual_vocab_size = transformed_df \
        .select(explode(col("filtered_tokens")).alias("word")) \
        .filter(length(col("word")) > 1) \
        .distinct() \
        .count()
    print(f"--> Actual vocabulary size after preprocessing: {actual_vocab_size} unique terms.")
    
    return actual_vocab_size

def vector_search(spark, text, pipeline_model, transformed_df):
    """Tìm kiếm văn bản tương tự trong tập dữ liệu đã chuyển đổi"""
    
    # Tạo DataFrame cho văn bản truy vấn
    query_df = spark.createDataFrame([(text,)], ["text"])
    
    # Transform văn bản truy vấn
    query_transformed = pipeline_model.transform(query_df)
    query_vector = query_transformed.select("features").first()[0]

    # Lấy một số văn bản mẫu và tính similarity đơn giản
    sample_results = transformed_df \
        .select("text", "features") \
        .limit(5) \
        .collect()
    
    # Tạo kết quả với similarity score đơn giản (demo)
    results = []
    for row in sample_results:
        # Đơn giản hóa: sử dụng số feature chung làm similarity score
        similarity = len(set(query_vector.indices) & set(row["features"].indices)) / max(len(query_vector.indices), 1)
        results.append({
            "text": row["text"][:200] + "..." if len(row["text"]) > 200 else row["text"],
            "similarity": round(similarity, 4)
        })
    
    # Sắp xếp theo similarity (cao nhất trước)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return results

def save_results(transformed_df, fit_duration, transform_duration, actual_vocab_size, search_text, search_results):
    """Lưu kết quả và metrics"""
    os.makedirs("results", exist_ok=True)
    
    # Lưu DataFrame output trực tiếp
    with open("results/lab3_dataframe_output.txt", "w", encoding="utf-8") as f:
        # Capture DataFrame.show() output
        import sys
        from io import StringIO
        
        # Redirect stdout to capture show() output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        print("--- Complete DataFrame Structure ---")
        transformed_df.printSchema()
        print("\n--- Sample Data (5 rows) ---")
        transformed_df.select("text", "tokens", "filtered_tokens", "features").show(5, truncate=False)
        
        # Restore stdout and get captured content
        sys.stdout = old_stdout
        dataframe_content = captured_output.getvalue()
        
        # Write to file
        f.write(dataframe_content)
    
    print("Successfully wrote DataFrame output to results/lab3_dataframe_output.txt")
    
    # Lưu thông tin pipeline model (thay vì Parquet để tránh lỗi Hadoop trên Windows)
    pipeline_info_path = "results/lab3_pipeline_info.txt"
    with open(pipeline_info_path, "w", encoding="utf-8") as f:
        f.write("--- Pipeline Model Information ---\n")
        f.write("Pipeline stages:\n")
        f.write("1. RegexTokenizer\n")
        f.write("2. StopWordsRemover\n") 
        f.write("3. HashingTF\n")
        f.write("4. IDF\n")
        f.write(f"Total features: {NUM_FEATURES}\n")
        f.write(f"Processing limit: {LIMIT_DOCUMENTS} documents\n")
    
    print(f"Successfully wrote pipeline info to {pipeline_info_path}")

    # Lưu metrics
    with open("results/lab3_metrics.log", "w", encoding="utf-8") as f:
        f.write("--- Performance Metrics ---\n")
        f.write(f"Pipeline fitting duration: {fit_duration:.2f} seconds\n")
        f.write(f"Data transformation duration: {transform_duration:.2f} seconds\n")
        f.write(f"Actual vocabulary size (after preprocessing): {actual_vocab_size} unique terms\n")
        f.write(f"HashingTF numFeatures set to: {NUM_FEATURES}\n")
        if NUM_FEATURES < actual_vocab_size:
            f.write(f"Note: numFeatures ({NUM_FEATURES}) is smaller than actual vocabulary size ({actual_vocab_size}). Hash collisions are expected.\n")
    
    
    # Lưu kết quả pipeline dạng bảng
    results = transformed_df.select("text", "features").take(20)
    with open("results/lab3_pipeline_output.txt", "w", encoding="utf-8") as f:
        f.write("--- NLP Pipeline Output (First 20 results) ---\n\n")
        
        # Header của bảng
        f.write("+" + "-" * 5 + "+" + "-" * 102 + "+" + "-" * 30 + "+\n")
        f.write("| {:<3} | {:<100} | {:<28} |\n".format("No.", "Original Text (first 100 chars)", "TF-IDF Vector Summary"))
        f.write("+" + "-" * 5 + "+" + "-" * 102 + "+" + "-" * 30 + "+\n")
        
        # Dữ liệu trong bảng
        for i, row in enumerate(results, 1):
            text = row["text"]
            features = row["features"]
            
            # Cắt text để fit trong bảng
            text_display = text[:97] + "..." if len(text) > 100 else text
            
            # Tóm tắt vector (số dimensions khác 0)
            vector_summary = f"Vec({features.size}, {len(features.indices)} non-zero)"
            
            f.write("| {:<3} | {:<100} | {:<28} |\n".format(i, text_display, vector_summary))
        
        f.write("+" + "-" * 5 + "+" + "-" * 102 + "+" + "-" * 30 + "+\n")
        
        # Chi tiết vector cho 5 records đầu
        f.write("\n\n--- Detailed TF-IDF Vectors (First 5 records) ---\n\n")
        for i, row in enumerate(results[:5], 1):
            text = row["text"]
            features = row["features"]
            f.write(f"Record {i}:\n")
            f.write(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}\n")
            f.write(f"Vector Size: {features.size}\n")
            f.write(f"Non-zero Elements: {len(features.indices)}\n")
            f.write(f"Top 10 Feature Indices: {features.indices[:10].tolist()}\n")
            f.write(f"Top 10 Feature Values: {[round(v, 4) for v in features.values[:10]]}\n")
            f.write("-" * 80 + "\n\n")
    
    print("Successfully wrote 20 results to results/lab3_pipeline_output.txt")

    print("Successfully wrote metrics to results/lab3_metrics.log")


    #Lưu kết quả vector search
    with open("results/lab3_vector_search.txt", "w", encoding="utf-8") as f:
        f.write("--- Vector Search Results ---\n\n")
        f.write(f"Search Text: {search_text}\n\n")
        for result in search_results:
            f.write(f"Text: {result['text']}\n")
            f.write(f"Similarity: {result['similarity']}\n")
            f.write("-" * 40 + "\n")
    print("Successfully wrote vector search results to results/lab3_vector_search.txt")

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
        transformed_df, pipeline_model, fit_duration, transform_duration = fit_and_transform_pipeline(pipeline, df)
        
        # 5. Phân tích từ vựng
        actual_vocab_size = analyze_vocabulary(transformed_df)
        
        # 6. Hiển thị kết quả mẫu
        print("\nSample of transformed data:")
        transformed_df.select("text", "features").show(5, truncate=False)
        
        # 7. Demo vector search
        text = input("\nInput text: ")
        search_results = vector_search(spark, text, pipeline_model, transformed_df)
        
        # 8. Lưu kết quả
        save_results(transformed_df, fit_duration, transform_duration, actual_vocab_size, text, search_results)
            
    finally:
            spark.stop()
            print("Spark Session stopped.")

if __name__ == "__main__":
    main()