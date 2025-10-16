# Lab 3: Pipeline NLP với PySpark và Tìm kiếm Vector
**Nguyễn Thùy Trang - 22000128**

## 1. Các bước triển khai chi tiết

### 1.1 Chuẩn bị môi trường
**Bước 1: Cài đặt các thư viện cần thiết**
```bash
pip install pyspark
```

**Bước 2: Thiết lập cấu trúc dự án**
```
src/
├── spark/
│   └── lab3_pyspark.py # Triển khai pipeline chính
data/
└── c4-train.00000-of-01024-30K.json.gz # Bộ dữ liệu nén
results/ # Thư mục kết quả đầu ra
```

### 1.2 Triển khai Pipeline NLP (4 giai đoạn)

**Bước 1: Khởi tạo phiên Spark**
**Bước 2: Tải dữ liệu nén**
**Bước 3: Tạo Pipeline NLP**
**Bước 4: Triển khai tìm kiếm Vector**

### 1.3 Cấu hình hệ thống
- **Nguồn dữ liệu**: `data/c4-train.00000-of-01024-30K.json.gz`
- **Giới hạn tài liệu**: 2,000 bản ghi
- **Số chiều HashingTF**: 20,000 chiều

## 2. Cách chạy code và ghi log kết quả

### 2.1 Lệnh chạy
```bash
# Từ thư mục gốc của dự án
python src/spark/lab3_pyspark.py
```

### 2.2 Quy trình thực hiện 
1. Tải dữ liệu JSON nén (2,000 bản ghi)
2. Tạo và huấn luyện pipeline NLP (4 giai đoạn)
3. Biến đổi dữ liệu thành vector TF-IDF
4. Thực hiện tìm kiếm vector với đầu vào từ người dùng
5. Lưu kết quả và ghi log


### 2.3 Kết quả
- File output: **`results/log3.txt`**: 
  - Cấu hình và chỉ số hiệu suất
  - Văn bản truy vấn đầu vào đầy đủ
  - Kết quả tìm kiếm vector với điểm tương tự
  - Mẫu vector đặc trưng dạng (số chiều,[indexes có tf-idf score != 0], [tf-idf scores] )
  
## 3. Giải thích kết quả thu được

### 3.1 Chỉ số hiệu suất pipeline
- **Thời gian huấn luyện pipeline**: 2.32 giây - Nhanh do bộ dữ liệu vừa phải
- **Thời gian biến đổi**: 1.02 giây - Hiệu quả nhờ tối ưu hóa Spark
- **Tổng thời gian xử lý**: 15.22 giây (không bao gồm I/O)
- **Tốc độ xử lý**: ~577 bản ghi/giây trong giai đoạn biến đổi

### 3.2 Phân tích từ vựng và không gian đặc trưng
- **Kích thước từ vựng thực tế**: 39,585 từ duy nhất
- **Không gian đặc trưng HashingTF**: 20,000 chiều
- **Tỷ lệ xung đột hash**: ~49.5%
- **Ý nghĩa**: Xung đột hash xảy ra nhưng vẫn trong mức chấp nhận được


### 3.2 Hiệu suất tìm kiếm
- **Thời gian tìm kiếm**: 11.49 giây
- **Số tài liệu so sánh**: 100
- **Số kết quả trả về**: 5

### 3.3 Kết quả tìm kiếm Vector
| Thứ hạng | Điểm tương tự | Xem trước tài liệu |
|----------|---------------|-------------------|
| 1 | 0.9538 | Lớp học BBQ cho người mới bắt đầu tại Missoula! Bạn có muốn trở nên giỏi hơn trong việc làm BBQ ngon?... |
| 2 | 0.0602 | Công ty chuyển nhà & Dịch vụ chuyển nhà tại Randallsville, New York cho mọi dịch vụ chuyển nhà... |
| 3 | 0.0511 | Học những kiến thức cơ bản về Flash khi chúng tôi hướng dẫn bạn qua phần giới thiệu toàn diện này... |
| 4 | 0.0500 | Liệu bạn có thể hiểu được cách có thể đưa những điều này đi xa hơn trong cuộc sống... |
| 5 | 0.0395 | Chủ đề này chứa 1 phản hồi, có 2 tiếng nói. Cập nhật lần cuối bởi Raam Dev 4 năm, 11 tháng trước... |

### 3.4 Phân tích vector đặc trưng
**Ví dụ tài liệu 1 (Lớp học BBQ)**:
- **Số chiều vector**: 20,000 
- **Phần tử khác không**: 62/20,000 
- **Giá trị TF-IDF hàng đầu**: [15.5513, 2.7339, 3.2576, ...] 
--> Các từ như "bbq", "class", "beginners" có trọng số cao


## 4. Đánh giá kết quả

### 4.1 Độ chính xác tìm kiếm Vector
- **Điểm tương tự cao nhất**: 0.9538 cho tài liệu khớp chính xác
- **Hiệu suất**: Tìm kiếm vector hoạt động tốt với các tài liệu tương tự
- **Độ tương tự Jaccard**: Phương pháp hiệu quả cho độ tương tự văn bản

### 4.2 Hiệu suất pipeline
- **Tốc độ xử lý**: ~577 bản ghi/giây trong giai đoạn biến đổi
- **Khả năng mở rộng**: Pipeline có thể xử lý bộ dữ liệu lớn hơn
- **Hiệu quả bộ nhớ**: Sử dụng vector thưa tiết kiệm bộ nhớ


## 5. Công nghệ sử dụng
- PySpark 3.x
- MLlib

