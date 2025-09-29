// Lab 3 - Data Loading Stage
// Nguyễn Thùy Trang - 22000128

import org.apache.spark.sql.SparkSession

object Lab3DataLoading {
	def main(args: Array[String]): Unit = {
		// 1. Khởi tạo SparkSession
		val spark = SparkSession.builder()
			.appName("Lab3 Data Loading")
			.master("local[*]")
			.getOrCreate()

		// 2. Đọc file JSON nén gzip
		val filePath = "c4-train.00000-of-01024-30K.json.gz"
		val df = spark.read.json(filePath)

		// 3. Lấy 1000 bản ghi đầu tiên để tăng tốc độ thực nghiệm
		val dfSample = df.limit(1000)

		// 4. Hiển thị schema và một số dòng dữ liệu mẫu
		dfSample.printSchema()
		dfSample.show(5, truncate = false)

		// 5. Kết thúc SparkSession
		spark.stop()
	}
}
