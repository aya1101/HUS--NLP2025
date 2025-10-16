# Laboratory Assignments NLP
Nguyễn Thùy Trang - 22000128
## ⊠ Các bước triển khai
1. Cài đặt Python (khuyến nghị >=3.8).
2. Cài đặt các thư viện cần thiết bằng lệnh:
   ```bash
   pip install -r requirements.txt
   ```
3. Đảm bảo các file dữ liệu (UD_English-EWT) đã có trong thư mục dự án.
4. Kiểm tra lại cấu trúc thư mục:
   - src/
   - test/
   - results/
   - UD_English-EWT/
   - requirements.txt
   - README.md

## Cách chạy code và ghi log kết quả
- Để chạy thử nghiệm và ghi log kết quả, sử dụng lệnh:
  ```bash
  python -m test.test_lab1 > results/log1.txt
  python -m test.test_lab2 > results/log2.txt
  ```
- Kết quả được ghi vào thư mục `results/` trong thư mục gốc.

##  Giải thích các kết quả thu được và nhận xét:
- Giải thích kết quả thu được và nhận xét của các lab được tổng hợp trong `results/Notes.md`

### Ví dụ kết quả thực tế
Toàn bộ kết quả test của các lab được lưu trong trong folder `results/`.

## Khó khăn gặp phải và cách giải quyết
- Lỗi đường dẫn ;-;
## • Model tạo sẵn, prompt
- Không sử dụng model tạo sẵn ngoài các tokenizer tự cài đặt trong thư mục `src/preprocessing`.

## Chạy PySpark Word2Vec (lab4 task)

Nếu muốn huấn luyện Word2Vec trên tập dữ liệu lớn bằng PySpark, dùng script sau:

PowerShell (ví dụ):
```powershell
& ".\.venv\Scripts\Activate.ps1";
python src/spark/lab4_task4_pyspark.py --input <path/to/jsonl_or_dir> --field text --output models/word2vec_spark --vectorSize 100 --minCount 5
```

Lưu ý:
- Cài PySpark: `pip install pyspark` trong venv.
- Trên Windows cần Java (JAVA_HOME) tương thích.
- Script sẽ ghi log mẫu vào `results/word2vec_sample.txt` và lưu model Spark vào thư mục `models/word2vec_spark`.
