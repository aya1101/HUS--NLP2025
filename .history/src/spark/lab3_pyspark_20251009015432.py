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
DATA_PATH = "data/c4-train.00000-of-01024-30K.json.gz"  # Compressed JSON file
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
    """Load compressed JSON file into Spark DataFrame"""
    print(f"Loading compressed JSON file: {data_path}")
    
    # Read compressed JSON file
    df = spark.read.json(data_path)
    
    # Filter out records with empty or very short text
    df = df.filter(F.length(F.col("text")) > 50).limit(limit)
    
    record_count = df.count()
    print(f"Successfully loaded {record_count:,} records (limited to {limit:,}).")
    
    # Show schema and sample data
    print("\nDataFrame Schema:")
    df.printSchema()
    print("\nSample of initial DataFrame:")
    df.select("text").show(3, truncate=True)
    
    return df

def create_pipeline():
    """Create Spark ML Pipeline for text preprocessing and vectorization"""
    print("\nCreating Spark ML Pipeline for Lab 17:")
    
    # Stage 1: Tokenization using RegexTokenizer
    print("1. RegexTokenizer - Tokenizing text into individual words")
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="tokens",
        pattern="\\W+"  # Split on non-word characters
    )
    
    # Stage 2: Stop Words Removal
    print("2. StopWordsRemover - Removing common low-information words")
    stop_words_remover = StopWordsRemover(
        inputCol="tokens",
        outputCol="filtered_tokens"
    )
    
    # Stage 3: Term Frequency using HashingTF
    print(f"3. HashingTF - Transform text into term frequency vectors ({NUM_FEATURES} features)")
    hashing_tf = HashingTF(
        inputCol="filtered_tokens",
        outputCol="raw_features",
        numFeatures=NUM_FEATURES
    )
    
    # Stage 4: Inverse Document Frequency
    print("4. IDF - Apply inverse document frequency weighting")
    idf = IDF(
        inputCol="raw_features",
        outputCol="features"
    )

    # Create Pipeline with 4 stages (no normalization for Lab 17)
    pipeline = Pipeline(stages=[tokenizer, stop_words_remover, hashing_tf, idf])
    print("Pipeline created successfully with 4 stages.\n")

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
    print(f"Pipeline fitting completed in {fit_duration:.2f} seconds")
    
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
    print(f"Data transformation of {transform_count:,} records completed in {transform_duration:.2f} seconds")
    
    # Performance summary
    total_pipeline_time = fit_duration + transform_duration
    print(f"\nPipeline Performance Summary:")
    print(f"   Fitting time: {fit_duration:.2f}s")
    print(f"   Transform time: {transform_duration:.2f}s")
    print(f"   Total pipeline time: {total_pipeline_time:.2f}s")
    print(f"   Records processed: {transform_count:,}")
    print(f"   Processing rate: {transform_count/total_pipeline_time:.1f} records/second")
    
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
    
    print(f"Vocabulary analysis completed in {vocab_duration:.2f} seconds")
    print(f"Vocabulary Statistics:")
    print(f"   Actual vocabulary size: {actual_vocab_size:,} unique terms")
    print(f"   HashingTF feature space: {NUM_FEATURES:,} dimensions")
    
    if NUM_FEATURES < actual_vocab_size:
        collision_rate = (actual_vocab_size - NUM_FEATURES) / actual_vocab_size * 100
        print(f"   Hash collision rate: ~{collision_rate:.1f}% (expected)")
    else:
        print(f"   Hash collision rate: 0% (feature space sufficient)")
    
    return actual_vocab_size, vocab_duration

def find_top_k_similar_documents(spark, query_text, pipeline_model, transformed_df, k=TOP_K_SIMILAR):
    """Tìm kiếm top K văn bản tương tự sử dụng cosine similarity"""
    print("\n" + "="*60)
    print(f"TOP {k} SIMILAR DOCUMENTS SEARCH")
    print("="*60)
    
    search_start_time = time.time()
    
    # Transform query text
    print(f"Query: '{query_text}'")
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
    
    print(f"Search completed in {search_duration:.2f} seconds")
    print(f"Top {k} Most Similar Documents:")
    
    for i, result in enumerate(top_k_results, 1):
        text_preview = result["text"][:100] + "..." if len(result["text"]) > 100 else result["text"]
        print(f"\n   {i}. Similarity: {result['similarity']:.4f}")
        print(f"      Text: {text_preview}")
    
    return top_k_results, search_duration

def save_pipeline_results(transformed_df, start_time, end_time, fit_duration, transform_duration, vocab_duration, actual_vocab_size):
    """Save pipeline results and log information according to Lab 17 requirements"""
    
    # Create directories
    os.makedirs("results", exist_ok=True)
    os.makedirs("log", exist_ok=True)
    
    # 1. Save feature vectors to results/lab17_pipeline_output.txt
    print("Saving feature vectors to results/lab17_pipeline_output.txt...")
    
    output_file = "results/lab17_pipeline_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("LAB 17 - SPARK ML PIPELINE OUTPUT\n")
        f.write("Nguyen Thuy Trang - 22000128\n")
        f.write("=" * 80 + "\n\n")
        
        # Configuration information
        f.write("CONFIGURATION:\n")
        f.write(f"Data source: {DATA_PATH}\n")
        f.write(f"Document limit: {LIMIT_DOCUMENTS:,}\n")
        f.write(f"HashingTF features: {NUM_FEATURES:,}\n\n")
        
        # Pipeline stages
        f.write("PIPELINE STAGES:\n")
        f.write("1. RegexTokenizer - Text tokenization\n")
        f.write("2. StopWordsRemover - Stop word removal\n")
        f.write("3. HashingTF - Term frequency vectorization\n")
        f.write("4. IDF - Inverse document frequency weighting\n\n")
        
        # Vocabulary statistics
        f.write("VOCABULARY ANALYSIS:\n")
        f.write(f"Actual vocabulary size: {actual_vocab_size:,} unique terms\n")
        f.write(f"HashingTF feature space: {NUM_FEATURES:,} dimensions\n")
        
        if NUM_FEATURES < actual_vocab_size:
            collision_rate = (actual_vocab_size - NUM_FEATURES) / actual_vocab_size * 100
            f.write(f"Hash collision rate: ~{collision_rate:.1f}%\n\n")
        else:
            f.write("Hash collision rate: 0% (sufficient feature space)\n\n")
        
        # Sample feature vectors
        f.write("FEATURE VECTORS (Sample):\n")
        f.write("-" * 50 + "\n")
        
        sample_vectors = transformed_df.select("text", "features").limit(10).collect()
        for i, row in enumerate(sample_vectors, 1):
            text = row["text"]
            features = row["features"]
            
            f.write(f"Record {i}:\n")
            f.write(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}\n")
            f.write(f"Vector size: {features.size}\n")
            f.write(f"Non-zero elements: {len(features.indices)}\n")
            f.write(f"Feature indices: {features.indices[:10].tolist()}\n")
            f.write(f"Feature values: {[round(v, 4) for v in features.values[:10]]}\n")
            f.write("-" * 50 + "\n")
    
    # 2. Log process information to log directory
    from datetime import datetime
    
    log_file = f"log/lab17_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    print(f"Logging process information to {log_file}...")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("LAB 17 - SPARK ML PIPELINE PROCESS LOG\n")
        f.write("=" * 80 + "\n\n")
        
        # Job information
        f.write("JOB INFORMATION:\n")
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total duration: {(end_time - start_time).total_seconds():.2f} seconds\n\n")
        
        # Performance metrics
        f.write("PERFORMANCE METRICS:\n")
        f.write(f"Pipeline fitting: {fit_duration:.2f} seconds\n")
        f.write(f"Data transformation: {transform_duration:.2f} seconds\n")
        f.write(f"Vocabulary analysis: {vocab_duration:.2f} seconds\n\n")
        
        # Data processing information
        f.write("DATA PROCESSING:\n")
        f.write(f"Source file: {DATA_PATH}\n")
        f.write(f"Records processed: {LIMIT_DOCUMENTS:,}\n")
        f.write(f"Vocabulary size: {actual_vocab_size:,} unique terms\n")
        f.write(f"Feature dimensions: {NUM_FEATURES:,}\n\n")
        
        # Success status
        f.write("JOB STATUS: COMPLETED SUCCESSFULLY\n")
        f.write("No errors encountered during processing.\n")
    
    print(f"Results saved to: {output_file}")
    print(f"Process log saved to: {log_file}")
    
    return output_file, log_file

def main():
    """
    Main function implementing Lab 17 requirements:
    1. Read compressed JSON file into Spark DataFrame
    2. Perform text preprocessing (tokenization + stop word removal)
    3. Vectorize data using HashingTF and IDF
    4. Save results to results/lab17_pipeline_output.txt
    5. Log process to log/ directory
    """
    
    from datetime import datetime
    start_time = datetime.now()
    
    print("=" * 80)
    print("LAB 17 - SPARK ML PIPELINE FOR TEXT VECTORIZATION")
    print("Nguyen Thuy Trang - 22000128")
    print("=" * 80)
    print(f"Configuration:")
    print(f"   Document limit: {LIMIT_DOCUMENTS:,}")
    print(f"   Feature dimensions: {NUM_FEATURES:,}")
    print(f"   Data source: {DATA_PATH}")
    print(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Initialize Spark Session
    spark = create_spark_session()
    
    try:
        # Step 1: Read compressed JSON data
        print("\nSTEP 1: Loading compressed JSON file into Spark DataFrame")
        df = load_data(spark, limit=LIMIT_DOCUMENTS)
        
        # Step 2: Create preprocessing and vectorization pipeline
        print("\nSTEP 2: Creating text preprocessing and vectorization pipeline")
        pipeline = create_pipeline()
        
        # Step 3: Fit pipeline and transform data
        print("\nSTEP 3: Fitting pipeline and transforming data")
        transformed_df, pipeline_model, fit_duration, transform_duration = fit_and_transform_pipeline(pipeline, df)
        
        # Step 4: Analyze vocabulary
        print("\nSTEP 4: Analyzing vocabulary")
        actual_vocab_size, vocab_duration = analyze_vocabulary(transformed_df)
        
        # Step 5: Display sample results
        print("\nSTEP 5: Displaying sample feature vectors")
        print("\nSample of feature vectors:")
        transformed_df.select("text", "features").show(3, truncate=True)
        
        # Step 6: Save results and log process
        end_time = datetime.now()
        print("\nSTEP 6: Saving results and logging process")
        output_file, log_file = save_pipeline_results(
            transformed_df, start_time, end_time, fit_duration, 
            transform_duration, vocab_duration, actual_vocab_size
        )
        
        # Final summary
        total_time = (end_time - start_time).total_seconds()
        print("\n" + "=" * 80)
        print("LAB 17 PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"FINAL SUMMARY:")
        print(f"   Total execution time: {total_time:.2f} seconds")
        print(f"   Documents processed: {df.count():,}")
        print(f"   Features created: {NUM_FEATURES:,}")
        print(f"   Vocabulary size: {actual_vocab_size:,}")
        print(f"   Results saved to: {output_file}")
        print(f"   Process log saved to: {log_file}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Pipeline execution failed. Check logs for details.")
        
        # Log error
        error_time = datetime.now()
        os.makedirs("log", exist_ok=True)
        error_log = f"log/lab17_error_{error_time.strftime('%Y%m%d_%H%M%S')}.log"
        with open(error_log, "w", encoding="utf-8") as f:
            f.write("LAB 17 - ERROR LOG\n")
            f.write("=" * 50 + "\n")
            f.write(f"Error time: {error_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Error message: {str(e)}\n")
        print(f"Error logged to: {error_log}")
        raise
        
    finally:
        spark.stop()
        print("\nSpark Session stopped.")
        print("Pipeline execution finished.")

if __name__ == "__main__":
    main()