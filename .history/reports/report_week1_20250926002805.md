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

- **Khó khăn**:
  - Gặp lỗi đường dẫn
  