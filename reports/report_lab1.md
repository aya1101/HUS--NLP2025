# Lab 1 & Lab 2: Tokenizer và CountVectorizer

## Cấu trúc repo
```
src/
├── core/
│   ├── interfaces.py              # Interface cho Tokenizer và Vectorizer
│   └── dataset_loaders.py         # DataLoader cho UD English EWT
├── preprocessing/
│   ├── simple_tokenizer.py        # SimpleTokenizer implementation
│   ├── regex_tokenizer.py         # RegexTokenizer implementation
│   └── stopwords_remover.py       # StopWords remover
├── representations/
│   └── count_vectoriser.py        # CountVectorizer implementation
└── model/
    └── text_classifier.py         # Text classifier

test/
├── test_lab1.py                   # Test cho Tokenizers
└── test_lab2.py                   # Test cho CountVectorizer

data/
└── UD_English-EWT/               # Universal Dependencies dataset
    ├── en_ewt-ud-train.conllu
    ├── en_ewt-ud-dev.conllu
    └── en_ewt-ud-test.conllu

results/
├── log1.txt                      # Kết quả Lab 1
└── log2.txt                      # Kết quả Lab 2
```

## Các bước triển khai

### Lab 1: Tokenizer
1. **Interface**: Định nghĩa abstract base class `Tokenizer` trong `core/interfaces.py`
2. **SimpleTokenizer**: Triển khai tokenizer đơn giản giữ lại dấu câu
3. **RegexTokenizer**: Triển khai tokenizer sử dụng regex loại bỏ dấu câu
4. **Testing**: Test trên ví dụ cơ bản và dataset UD English EWT

### Lab 2: CountVectorizer  
1. **Interface**: Định nghĩa abstract base class `Vectorizer` 
2. **CountVectorizer**: Triển khai vector hóa văn bản dựa trên tần suất từ
3. **Testing**: Test trên ví dụ cơ bản và dataset UD English EWT

## Cách chạy code
```bash
# Test Lab 1 - Tokenizers
python test/test_lab1.py

# Test Lab 2 - CountVectorizer  
python test/test_lab2.py
```

## Kết quả thực hiện

- **Lab 1**
  - Kết quả được lưu tại tệp `results/log1.txt`
  - `SimpleTokenizer` giữ lại dấu câu và các ký tự đặc biệt như dấu gạch ngang, dấu phẩy, dấu hai chấm.
  - `RegexTokenizer` tách các từ, loại bỏ dấu câu, giúp chuẩn hóa token cho các tác vụ NLP.
  - Số lượng token và nội dung token sẽ khác nhau tùy tokenizer, phản ánh chiến lược tách từ khác nhau.

- **Lab 2**
  - Kết quả được lưu tại tệp `results/log2.txt`
  - `CountVectorizer` chuyển mỗi văn bản thành một vector đếm số lần xuất hiện của từng từ trong từ vựng. Kết quả là ma trận tài liệu-từ vựng (document-term matrix), giúp biểu diễn văn bản dưới dạng số để phục vụ các bài toán học máy.
  - Từ vựng (vocabulary) được xây dựng từ toàn bộ corpus, mỗi từ được gán một chỉ số duy nhất. Các văn bản khác nhau sẽ có vector khác nhau tùy theo số lượng và loại từ xuất hiện.
  - Nếu tokenizer tách từ khác nhau, ma trận vector hóa cũng sẽ khác nhau, cho thấy tầm quan trọng của bước tiền xử lý.

## Giải thích kết quả

### So sánh Tokenizers:
- **SimpleTokenizer**: Giữ nguyên dấu câu như ".", ",", "!" làm token riêng biệt
  - Ưu điểm: Bảo toàn thông tin câu trúc văn bản
  - Nhược điểm: Vocabulary size lớn hơn, nhiều noise

- **RegexTokenizer**: Loại bỏ dấu câu, chỉ giữ lại từ và số
  - Ưu điểm: Vocabulary sạch hơn, phù hợp cho phân tích semantic
  - Nhược điểm: Mất thông tin cấu trúc câu

### CountVectorizer Performance:
- Trên ví dụ cơ bản: Tạo được sparse matrix với vocabulary nhỏ
- Trên UD dataset: Vocabulary lớn (~20K words), sparse matrix hiệu quả
- Memory usage: Sử dụng scipy sparse matrix để tiết kiệm bộ nhớ

## Khó khăn & giải pháp

### Khó khăn gặp phải:
1. **Lỗi đường dẫn**: Import modules từ các thư mục khác
   - **Giải pháp**: Sửa `sys.path` và sử dụng relative imports đúng cách

2. **Xử lý Unicode**: Dataset UD có các ký tự đặc biệt  
   - **Giải pháp**: Sử dụng encoding='utf-8' khi đọc file

3. **Memory issues**: CountVectorizer trên dataset lớn
   - **Giải pháp**: Sử dụng scipy.sparse matrix thay vì dense array

4. **Regex patterns**: Tối ưu regex cho RegexTokenizer
   - **Giải pháp**: Sử dụng `\b\w+\b` để tách từ hiệu quả


## Tài liệu tham khảo
- Scikit-learn CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- Python Regex Documentation: https://docs.python.org/3/library/re.html
- Scipy Sparse Matrices: https://docs.scipy.org/doc/scipy/reference/sparse.html

## Model/Tool sử dụng
- **Libraries**: Python standard library (re, os), numpy, scipy.sparse
- **Dataset**: Universal Dependencies English-EWT v2.7
  