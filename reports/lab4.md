# Lab 4: Phân loại văn bản

## Cấu trúc thư mục
```
data/
├── lab5/
│   ├── sent_all_cleaned.csv        # Dữ liệu đã gộp và làm sạch
│   ├── sent_train.csv              # Dữ liệu huấn luyện
│   └── sent_valid.csv              # Dữ liệu validation

src/
├── preprocessing/
│   └── regex_tokenizer.py          # Tokenizer với regex
├── representations/
│   └── count_vectoriser.py         # Count vectorizer
├── model/
│   └── text_classifier.py         # Text classifier model
└── spark/
    └── lab4/
        └── task3_lab4_sentiment_analysis.py  # PySpark sentiment analysis

test/
└── lab4/
    ├── task2_lab4_test.py          # Test cho baseline
    └── task4_lab4_advanced.py      # Test cho Word2Vec + PySpark

results/
└── lab4/
    ├── task1_2_results.txt         # Kết quả baseline
    ├── task2_train_valid_results.txt
    ├── task3_lab4_sentiment_analysis_results.txt
    └── task4.txt                   # Kết quả Word2Vec advanced
```

### Các bước triển khai 
- Chuẩn bị dữ liệu: gộp và làm sạch `sent_train.csv` + `sent_valid.csv` → `sent_all_cleaned.csv`.
- Tiền xử lý: loại URL/email, xóa ký tự đặc biệt, thu nhỏ chữ, chuẩn hóa khoảng trắng, loại hàng trống.
- Đặc trưng: CountVectorizer (baseline) hoặc Word2Vec / TF-IDF (nâng cao bằng PySpark).
- Huấn luyện: LogisticRegression cho baseline; Word2Vec + LogisticRegression/NaiveBayes trên PySpark cho nâng cao.
- Đánh giá: accuracy, precision/recall, F1, ma trận nhầm lẫn; lưu kết quả vào `results/lab4/`.

## Thực hiện
**Task 1-2**: Pipeline phân loại văn bản sử dụng LogisticRegression
- Dữ liệu: 
  - Dataset 1: 6 bình luận phim với nhãn nhị phân (tích cực/tiêu cực)
  - Dataset 2: Bộ dữ liệu comment twitter với 3 nhãn (bearish/ bullish/ neutral)
- Tiền xử lý: RegexTokenizer + CountVectorizer
- Mô hình: LogisticRegression với solver 'lbfgs'

**Task 3**: Pipeline phân tích cảm xúc bằng PySpark (TF-IDF + LogReg)
- Dữ liệu: đọc `sent_train.csv` (train) và `sent_valid.csv` (validation).
- Tiền xử lý: loại URL/email/ký tự đặc biệt bằng Spark SQL (`regexp_replace`), tokenize và loại stopwords bằng `Tokenizer` + `StopWordsRemover`.
- Biểu diễn: HashingTF → IDF (TF-IDF) để tạo feature vector bằng Spark ML.
- Mô hình: LogisticRegression (Spark ML) với các tham số điều chỉnh (ví dụ maxIter, regParam).
- Đánh giá: đo thời gian huấn luyện/đánh giá, tính Accuracy, F1 và ma trận nhầm lẫn; kết quả lưu vào `results/lab4/task3_lab4_sentiment_analysis_results.txt`.

**Task 4**: Nâng cao bằng embeddings và mô hình thay thế (Word2Vec + NaiveBayes/LogReg trên PySpark)
- Dữ liệu: dùng `sent_all_cleaned.csv` (đã gộp và tiền xử lý trước).
- Tiền xử lý: tiền xử lý tương tự trên Spark (lọc, regex, tokenize, stopwords).
- Biểu diễn: huấn luyện Word2Vec (Spark ML) để lấy vector trung bình cho mỗi câu; có thể so sánh hoặc kết hợp với TF-IDF.
- Mô hình: NaiveBayes (Multinomial) và/hoặc LogisticRegression trên feature embedding; xây pipeline đầy đủ với `Pipeline` của Spark ML.
- Đánh giá: đo thời gian huấn luyện, đánh giá trên tập test, lưu kết quả vào `results/lab4/task4.txt`; so sánh hiệu suất với baseline và phân tích sự khác biệt.

## Hướng dẫn chạy 
- Chạy test task 2 (baseline):
```powershell
& ".\.venv\Scripts\python.exe" test\lab4\task2_lab4_test.py
```

- Chạy Task 3 (PySpark):
```powershell
& ".\.venv\Scripts\python.exe" src\spark\lab4\task3_lab4_sentiment_analysis.py
```

- Chạy Task 4 (PySpark):
```powershell
& ".\.venv\Scripts\python.exe" test\lab4\task4_lab4_advanced.py
```

- Kết quả lưu trong `results/lab4/` (ví dụ `task1_2_results.txt`, `task2_train_valid_results.txt`, `task4.txt`).


## Kết quả

- Baseline (LogisticRegression trên tập mẫu nhỏ):
  - Accuracy: 0.500
  - F1-score: 0.333

- Mô hình nâng cao (Word2Vec + LogisticRegression, PySpark) — kết quả từ `results/lab4/task4.txt`:
  - Dữ liệu: 11,927 mẫu (class0=1,789; class1=2,398; class2=7,740)
  - Train/test split: 9,606 / 2,321
  - Accuracy: 0.6682
  - F1-score: 0.5726
  - Thời gian huấn luyện: ~6.9s; đánh giá: ~1.59s
**Thời gian huấn luyện**: 0.0040 giây
**Thời gian dự đoán**: 0.0010 giây



## Phân tích
- Mô hình nâng cao cho kết quả tốt hơn đáng kể vì đã dùng:
  - Dữ liệu thực lớn hơn giúp học được biểu diễn ngôn ngữ.
  - Word2Vec cung cấp embedding từ, giữ được thông tin ngữ nghĩa hơn BoW.
  - PySpark cho phép xử lý và huấn luyện nhanh trên tập lớn.

- Hạn chế còn tồn tại:
  - Dữ liệu không cân bằng (class2 lớn) gây thiên lệch dự đoán về lớp nhiều mẫu.
  - Thiếu tối ưu hyperparameter cho Word2Vec/LogReg; có thể cải thiện bằng tuning và cân bằng lớp.
---
Tóm lại: baseline trên dữ liệu rất nhỏ không đủ tin cậy; khi dùng dữ liệu thực và Word2Vec+LogReg (PySpark) hiệu suất tăng (Accuracy ~0.67, F1 ~0.57). Hướng tiếp theo: cân bằng lớp, tuning hyperparameters hoặc thử các mô hình mạnh hơn.

## Khó khăn & Giải pháp
- Lỗi import/module: đã sửa `sys.path` để đúng cấu trúc dự án.
- Xung đột thư viện (NumPy 2.x): đã hạ NumPy xuống 1.26.4.
- CSV có URL/quotes/nhãn bị sai kiểu: áp dụng tiền xử lý, ép kiểu label sang integer và drop NaN.
- PySpark ban đầu gặp lỗi kiểu dữ liệu: ép kiểu label (IntegerType) khi đọc và debug pipeline.

## Tài liệu tham khảo
- Scikit-learn documentation
- PySpark ML documentation (Word2Vec, NaiveBayes, Pipeline)
- Gensim / Word2Vec tutorials (tham khảo khi thử nghiệm embeddings)
- Codebase: `src/preprocessing/regex_tokenizer.py`, `src/representations/count_vectoriser.py`, `src/model/text_classifier.py`
- Dữ liệu: : https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
 
