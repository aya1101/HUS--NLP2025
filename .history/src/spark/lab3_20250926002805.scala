// Lab 3 - Data Loading Stage
// Nguyễn Thùy Trang - 22000128

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer}
import java.io.{File, PrintWriter}
object Lab3NLPPipeline {
	def main(args: Array[String]): Unit = {
		// 1. Khởi tạo SparkSession
		val spark = SparkSession.builder()
			.appName("Lab3 Data Loading")
			.master("local[*]")
			.getOrCreate()
		
		// UI
		import spark.implicits._
		println("Spark Session created successfully.")
		println(s"Spark UI available at http://localhost:4040")
		println("Pausing for 10 seconds to allow you to open the Spark UI...")
		Thread.sleep(10000)

		// 2. Đọc file JSON
		val filePath = "/data/c4-train.00000-of-01024-30K.json"
		val df = spark.read.json(filePath)
		val dfSample = df.limit(1000)

		dfSample.printSchema()
		dfSample.show(5, truncate = false)

		// 4. Tokenization: Sử dụng RegexTokenizer để tách từ
		val regexTokenizer = new RegexTokenizer()
					.setInputCol("text") 
					.setOutputCol("tokens")
					.setPattern("\\s+|[^\\w\\s]+")

		val tokenizedDF = regexTokenizer.transform(dfSample)
		tokenizedDF.select("text", "tokens").show(5, truncate = false)

		// 3. Stop Word Removal
		val remover = new StopWordsRemover()
					.setInputCol("tokens")
					.setOutputCol("filtered")

				val filteredDF = remover.transform(tokenizedDF)
				filteredDF.select("text", "filtered").show(5, truncate = false)

		// 4. TF-IDF Vectorization
		val hashingTF = new HashingTF()
			.setInputCol("filtered")
			.setOutputCol("rawFeatures")
			.setNumFeatures(10000)

		val featurizedDF = hashingTF.transform(filteredDF)

		val idf = new IDF()
			.setInputCol("rawFeatures")
			.setOutputCol("features")

		val idfModel = idf.fit(featurizedDF)
		val rescaledDF = idfModel.transform(featurizedDF)

		spark.stop()
	}
	
}
