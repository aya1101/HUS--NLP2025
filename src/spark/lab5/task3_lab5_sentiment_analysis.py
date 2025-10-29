import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import expr

# 1. Initialize Spark Session
spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()

# 2. Load Data

train_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lab5', 'sent_train.csv'))
valid_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lab5', 'sent_valid.csv'))

df_train = spark.read.option("header", "true").option("escape", '"').option("multiLine", "true").csv(train_path)
df_valid = spark.read.option("header", "true").option("escape", '"').option("multiLine", "true").csv(valid_path)

df_train = df_train.select("text", expr("try_cast(label as double) as label")).dropna(subset=["label"])
df_valid = df_valid.select("text", expr("try_cast(label as double) as label")).dropna(subset=["label"])


# 3. Build Preprocessing Pipeline
tokenizer = Tokenizer(inputCol="text", outputCol="words")
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=5000)
idf = IDF(inputCol="raw_features", outputCol="features")

# 4. Train the Model with timing
train_start = time.time()
lr = LogisticRegression(maxIter=100, regParam=0.01, featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])
model = pipeline.fit(df_train)
train_time = time.time() - train_start

# 5. Evaluate the Model with timing
eval_start = time.time()
predictions = model.transform(df_valid)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = f1_evaluator.evaluate(predictions)
eval_time = time.time() - eval_start

# Prepare output
output_lines = []
output_lines.append(f"Model training time: {train_time:.4f} seconds")
output_lines.append(f"Model evaluation time: {eval_time:.4f} seconds")
output_lines.append(f"Test Accuracy: {accuracy:.4f}")
output_lines.append(f"Test F1 Score: {f1:.4f}")

# Save to results folder
results_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'lab5')
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, 'task3_lab5_sentiment_analysis_results.txt')
with open(output_path, 'w', encoding='utf-8') as f:
	f.write('\n'.join(output_lines))

# Also print to console
for line in output_lines:
	print(line)