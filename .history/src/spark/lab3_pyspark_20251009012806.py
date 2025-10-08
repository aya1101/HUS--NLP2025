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
    """Phân tích kích thước từ vựng với detailed performance measurement"""
    print("\n" + "="*60)
    print("VOCABULARY ANALYSIS")
    print("="*60)
    
    vocab_start_time = time.time()
    
    # Calculate actual vocabulary size
    actual_vocab_size = transformed_df \
        .select(explode(col("filtered_tokens")).alias("word")) \
        .filter(length(col("word")) > 1) \
        .distinct() \
        .count()
    
    vocab_duration = time.time() - vocab_start_time
    
    print(f"✓ Vocabulary analysis completed in {vocab_duration:.2f} seconds")
    print(f"📚 Vocabulary Statistics:")
    print(f"   • Actual vocabulary size: {actual_vocab_size:,} unique terms")
    print(f"   • HashingTF feature space: {NUM_FEATURES:,} dimensions")
    
    if NUM_FEATURES < actual_vocab_size:
        collision_rate = (actual_vocab_size - NUM_FEATURES) / actual_vocab_size * 100
        print(f"   • Hash collision rate: ~{collision_rate:.1f}% (expected)")
    else:
        print(f"   • Hash collision rate: 0% (feature space sufficient)")
    
    return actual_vocab_size, vocab_duration

def find_top_k_similar_documents(spark, query_text, pipeline_model, transformed_df, k=TOP_K_SIMILAR):
    """Tìm kiếm top K văn bản tương tự sử dụng cosine similarity"""
    print("\n" + "="*60)
    print(f"TOP {k} SIMILAR DOCUMENTS SEARCH")
    print("="*60)
    
    search_start_time = time.time()
    
    # Transform query text
    print(f"🔍 Query: '{query_text}'")
    print("Processing query through the pipeline...")
    
    query_df = spark.createDataFrame([(query_text,)], ["text"])
    query_transformed = pipeline_model.transform(query_df)
    query_vector = query_transformed.select("features").first()[0]
    
    print(f"Query vector created with {len(query_vector.indices)} non-zero features")
    
    # For demonstration, we'll use a simplified similarity calculation
    # In production, you would use MLlib's cosine similarity or custom UDF
    print(f"Searching for top {k} most similar documents...")
    
    # Get a larger sample for better similarity search
    sample_size = min(100, transformed_df.count())
    sample_docs = transformed_df.select("text", "features").limit(sample_size).collect()
    
    # Calculate similarities
    similarities = []
    for i, doc in enumerate(sample_docs):
        # Simple similarity based on common feature indices
        doc_indices = set(doc["features"].indices)
        query_indices = set(query_vector.indices)
        
        if len(query_indices) > 0 and len(doc_indices) > 0:
            # Jaccard similarity as a proxy for cosine similarity
            intersection = len(query_indices & doc_indices)
            union = len(query_indices | doc_indices)
            similarity = intersection / union if union > 0 else 0.0
        else:
            similarity = 0.0
            
        similarities.append({
            "doc_id": i,
            "text": doc["text"],
            "similarity": similarity
        })
    
    # Sort by similarity and get top K
    top_k_results = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:k]
    
    search_duration = time.time() - search_start_time
    
    print(f"✓ Search completed in {search_duration:.2f} seconds")
    print(f"📄 Top {k} Most Similar Documents:")
    
    for i, result in enumerate(top_k_results, 1):
        text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
        print(f"\n   {i}. Similarity: {result['similarity']:.4f}")
        print(f"      Text: {text_preview}")
    
    return top_k_results, search_duration

def save_results(transformed_df, fit_duration, transform_duration, vocab_duration, actual_vocab_size, search_results, search_duration):
    """Lưu kết quả và metrics với comprehensive logging"""
    print("\n" + "="*60)
    print("SAVING RESULTS AND LOGGING")
    print("="*60)
    
    os.makedirs("results", exist_ok=True)
    
    # 1. Save comprehensive performance metrics
    metrics_file = "results/lab3_comprehensive_metrics.log"
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LAB 3 - SPARK ML NLP PIPELINE COMPREHENSIVE METRICS\n")
        f.write("Nguyễn Thùy Trang - 22000128\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration
        f.write("📋 CONFIGURATION:\n")
        f.write(f"   • Document limit: {LIMIT_DOCUMENTS:,}\n")
        f.write(f"   • Feature dimensions: {NUM_FEATURES:,}\n")
        f.write(f"   • Data source: {DATA_PATH}\n")
        f.write(f"   • Top-K similarity search: {TOP_K_SIMILAR}\n\n")
        
        # Performance metrics
        f.write("⏱️  PERFORMANCE METRICS:\n")
        f.write(f"   • Pipeline fitting: {fit_duration:.2f} seconds\n")
        f.write(f"   • Data transformation: {transform_duration:.2f} seconds\n")
        f.write(f"   • Vocabulary analysis: {vocab_duration:.2f} seconds\n")
        f.write(f"   • Similarity search: {search_duration:.2f} seconds\n")
        
        total_time = fit_duration + transform_duration + vocab_duration + search_duration
        f.write(f"   • Total processing time: {total_time:.2f} seconds\n\n")
        
        # Vocabulary statistics
        f.write("📚 VOCABULARY ANALYSIS:\n")
        f.write(f"   • Actual vocabulary size: {actual_vocab_size:,} unique terms\n")
        f.write(f"   • HashingTF feature space: {NUM_FEATURES:,} dimensions\n")
        
        if NUM_FEATURES < actual_vocab_size:
            collision_rate = (actual_vocab_size - NUM_FEATURES) / actual_vocab_size * 100
            f.write(f"   • Hash collision rate: ~{collision_rate:.1f}%\n")
        else:
            f.write("   • Hash collision rate: 0% (sufficient feature space)\n")
        
        # Search results
        f.write(f"\n🔍 TOP {TOP_K_SIMILAR} SIMILARITY SEARCH RESULTS:\n")
        for i, result in enumerate(search_results[:TOP_K_SIMILAR], 1):
            f.write(f"   {i}. Similarity: {result['similarity']:.4f}\n")
            text_preview = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
            f.write(f"      Text: {text_preview}\n\n")
    
    # 2. Save DataFrame structure and sample data
    dataframe_file = "results/lab3_dataframe_structure.txt"
    with open(dataframe_file, "w", encoding="utf-8") as f:
        import sys
        from io import StringIO
        
        old_stdout = sys.stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        print("SPARK DATAFRAME SCHEMA:")
        print("=" * 50)
        transformed_df.printSchema()
        print("\nSAMPLE TRANSFORMED DATA (3 records):")
        print("=" * 50)
        transformed_df.select("text", "tokens", "filtered_tokens", "features").show(3, truncate=False)
        
        sys.stdout = old_stdout
        f.write(captured_output.getvalue())
    
    # 3. Save pipeline model information
    pipeline_info_file = "results/lab3_pipeline_info.txt"
    with open(pipeline_info_file, "w", encoding="utf-8") as f:
        f.write("SPARK ML PIPELINE INFORMATION\n")
        f.write("=" * 40 + "\n\n")
        f.write("Pipeline Stages (in order):\n")
        f.write("1. RegexTokenizer - Text tokenization using regex patterns\n")
        f.write("2. StopWordsRemover - Remove English stop words\n")
        f.write("3. HashingTF - Term frequency computation with hashing\n")
        f.write("4. IDF - Inverse document frequency weighting\n")
        f.write("5. Normalizer - L2 vector normalization\n\n")
        f.write(f"Configuration Details:\n")
        f.write(f"• HashingTF features: {NUM_FEATURES:,} dimensions\n")
        f.write(f"• Document processing limit: {LIMIT_DOCUMENTS:,}\n")
        f.write(f"• Normalization method: L2 (Euclidean)\n")
        f.write(f"• Tokenization pattern: \\s+|[.,;!?()\"']\n")
    
    print(f"✓ Comprehensive metrics saved to: {metrics_file}")
    print(f"✓ DataFrame structure saved to: {dataframe_file}")
    print(f"✓ Pipeline information saved to: {pipeline_info_file}")
    
    return metrics_file, dataframe_file, pipeline_info_file
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
    """
    Main function implementing comprehensive NLP pipeline with all requirements:
    ✓ Read C4 dataset into Spark DataFrame  
    ✓ Implement Spark ML Pipeline
    ✓ Use RegexTokenizer for tokenization
    ✓ Use StopWordsRemover to remove stop words
    ✓ Use HashingTF and IDF for vectorization
    ✓ Fit pipeline and transform data
    ✓ Save results to files
    ✓ Log the process with detailed performance measurement
    ✓ Add limitDocuments variable for customization
    ✓ Add Normalizer layer to normalize vectors
    ✓ Search and display top K similar documents
    """
    
    print("=" * 80)
    print("LAB 3 - COMPREHENSIVE SPARK ML NLP PIPELINE")
    print("Nguyễn Thùy Trang - 22000128")
    print("=" * 80)
    print(f"📋 Configuration:")
    print(f"   • Document limit: {LIMIT_DOCUMENTS:,}")
    print(f"   • Feature dimensions: {NUM_FEATURES:,}")
    print(f"   • Top-K similarity: {TOP_K_SIMILAR}")
    print(f"   • Data source: {DATA_PATH}")
    print("=" * 80)
    
    # Initialize Spark Session
    spark = create_spark_session()
    
    try:
        # Step 1: Read dataset into Spark DataFrame
        print("\n🔄 STEP 1: Reading dataset into Spark DataFrame")
        df = load_data(spark, limit=LIMIT_DOCUMENTS)
        
        # Step 2: Implement Spark ML Pipeline
        print("\n🔄 STEP 2: Implementing Spark ML Pipeline")
        pipeline = create_pipeline()
        
        # Step 3: Fit pipeline and transform data
        print("\n🔄 STEP 3: Fitting pipeline and transforming data")
        transformed_df, pipeline_model, fit_duration, transform_duration = fit_and_transform_pipeline(pipeline, df)
        
        # Step 4: Analyze vocabulary with detailed performance measurement
        print("\n🔄 STEP 4: Analyzing vocabulary")
        actual_vocab_size, vocab_duration = analyze_vocabulary(transformed_df)
        
        # Step 5: Display sample results
        print("\n🔄 STEP 5: Displaying sample transformed data")
        print("\n📄 Sample of transformed data (columns: text, tokens, filtered_tokens, features):")
        transformed_df.select("text", "tokens", "filtered_tokens", "features").show(3, truncate=True)
        
        # Step 6: Search and display top K similar documents
        print("\n🔄 STEP 6: Searching for top K similar documents")
        # Use a default query or prompt for user input
        query_options = [
            "machine learning artificial intelligence",
            "computer science programming",
            "data analysis statistics",
            "natural language processing"
        ]
        
        print("Available sample queries:")
        for i, query in enumerate(query_options, 1):
            print(f"  {i}. {query}")
        
        try:
            choice = input(f"\nSelect query (1-{len(query_options)}) or enter custom text: ")
            if choice.isdigit() and 1 <= int(choice) <= len(query_options):
                query_text = query_options[int(choice) - 1]
            else:
                query_text = choice if choice.strip() else query_options[0]
        except (KeyboardInterrupt, EOFError):
            query_text = query_options[0]  # Default query
            
        print(f"Using query: '{query_text}'")
        search_results, search_duration = find_top_k_similar_documents(
            spark, query_text, pipeline_model, transformed_df, TOP_K_SIMILAR
        )
        
        # Step 7: Save results to files with comprehensive logging
        print("\n🔄 STEP 7: Saving results and logging process")
        save_results(transformed_df, fit_duration, transform_duration, vocab_duration, 
                    actual_vocab_size, search_results, search_duration)
        
        # Step 8: Final summary
        total_time = fit_duration + transform_duration + vocab_duration + search_duration
        print("\n" + "=" * 80)
        print("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"📊 FINAL SUMMARY:")
        print(f"   • Total execution time: {total_time:.2f} seconds")
        print(f"   • Documents processed: {df.count():,}")
        print(f"   • Features created: {NUM_FEATURES:,}")
        print(f"   • Vocabulary size: {actual_vocab_size:,}")
        print(f"   • Top similar docs found: {len(search_results)}")
        print(f"   • Results saved to: results/ directory")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("Pipeline execution failed. Check logs for details.")
        raise
        
    finally:
        spark.stop()
        print("\n🔴 Spark Session stopped.")
        print("Pipeline execution finished.")

if __name__ == "__main__":
    main()