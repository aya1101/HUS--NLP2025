# 🔬 Natural Language Processing Laboratory Assignments
> **Sinh viên:** Nguyễn Thùy Trang - 22000128  
> **Khóa học:** HUS NLP 2025  
> **Kho lưu trữ:** [HUS--NLP2025](https://github.com/aya1101/HUS--NLP2025)

---

## Tổng quan dự án

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) được triển khai từ cơ bản đến nâng cao, bao gồm các kỹ thuật xử lý văn bản, word embeddings, classification, và deep learning.

---

## Cấu trúc dự án

```
📁 HW1_backup/
├── 📂 src/                    # Source code modules
│   ├── core/                  # Core interfaces & dataset loaders
│   ├── preprocessing/         # Text tokenizers & stopwords
│   ├── representations/       # Vectorizers & embedders
│   ├── model/                 # ML classifiers
│   └── spark/                 # PySpark implementations
├── 📂 test/                   # Unit tests & lab demos
├── 📂 data/                   # Datasets (UD, sentiment, NLU)
├── 📂 reports/                # Lab reports & notebooks
├── 📂 results/                # Experimental outputs
├── 📂 models/                 # Trained models
└── 📄 README.md               # Project overview
```

### Nội dung các lab

- **Lab 1 — Tách từ và biểu diễn văn bản bằng vector thưa:** Thực hiện tokenization, normalization, loại bỏ stopwords và tiền xử lý đầu vào. 

- **Lab 2 — Biểu diễn văn bản:** Vector hóa văn bản bằng các phương pháp truyền thống (TF-IDF, Bag-of-Words). 

- **Lab 3 — Word Embeddings:** Huấn luyện và sử dụng embedding (Word2Vec, GloVe) và triển khai trên PySpark cho tập lớn. 

- **Lab 4 — Phân loại văn bản:** Xây dựng và đánh giá các bộ phân loại cho bài toán phân cực cảm xúc. 

- **Lab 5 — RNNs và các bài toán:** Thiết kế mô hình học sâu cho phân tích cảm xúc, huấn luyện và tinh chỉnh mạng nơ-ron. 

- **Lab 6 — Transformers và các bài toán:** Mô hình tuần tự như RNN/LSTM cho phân loại ý định và các bài toán nâng cao khác. 

---

## Cài đặt và triển khai

### Yêu cầu hệ thống
- **Python:** >= 3.8
- **RAM:** 8GB+ (cho word embeddings)
- **Storage:** 3GB+ (models + datasets)
- **OS:** Windows/Linux/macOS

### Bước 1: Cài đặt môi trường
```bash
# Clone repository
git clone https://github.com/aya1101/HUS--NLP2025.git
cd HW

# Tạo virtual environment (khuyến nghị)
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### Bước 2: Kiểm tra cấu trúc
```bash
# Verify project structure
ls src/ test/ data/ results/
```

---

## Hướng dẫn chạy code
Chạy code theo hướng dẫn trong từng report.