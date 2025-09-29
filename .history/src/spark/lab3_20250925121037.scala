// Lab 3 - Data Loading Stage
// Nguyễn Thùy Trang - 22000128

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover}
object Lab3DataLoading {
	def dataLoading(args: Array[String]): Unit = {
		// 1. Khởi tạo SparkSession
		val spark = SparkSession.builder()
			.appName("Lab3 Data Loading")
			.master("local[*]")
			.getOrCreate()

		// 2. Đọc file JSON
		val filePath = "c4-train.00000-of-01024-30K.json"
		val df = spark.read.json(filePath)
		val dfSample = df.limit(1000)

		dfSample.printSchema()
		dfSample.show(5, truncate = false)

				// 4. Tokenization: Sử dụng RegexTokenizer để tách từ
				val regexTokenizer = new RegexTokenizer()
					.setInputCol("text") // Giả sử cột chứa văn bản tên là "text"
					.setOutputCol("tokens")
					.setPattern("\\w+|[^\\w\\s]+") // tách từ và dấu câu

				val tokenizedDF = regexTokenizer.transform(dfSample)
				tokenizedDF.select("text", "tokens").show(5, truncate = false)

				// 5. Stop Word Removal: Loại bỏ stop words
				val remover = new StopWordsRemover()
					.setInputCol("tokens")
					.setOutputCol("filtered")

				val filteredDF = remover.transform(tokenizedDF)
				filteredDF.select("text", "filtered").show(5, truncate = false)

				// 6. Kết thúc SparkSession
				spark.stop()
	}
	
}
