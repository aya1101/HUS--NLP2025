# Lab 6 - Part 2: So sánh các phương pháp phân loại Intent

Trong bài lab này, chúng tôi đã thực hiện và so sánh 4 phương pháp khác nhau để phân loại intent từ dữ liệu HWU (Harbin Institute of Technology and University of Cambridge) dataset.

## Source code
- **File thực thi**: `src\lab6\lab6_part2.ipynb`  
- **Cấu trúc notebook**:
    + **Task 0** (Cell 1-6): Thiết lập môi trường, giải nén và load dữ liệu HWU
    + **Task 1** (Cell 7-9): TF-IDF + Logistic Regression (baseline)
    + **Task 2** (Cell 10-21): Word2Vec averaging + Dense Neural Network 
    + **Task 3** (Cell 22-31): Pre-trained Word2Vec Embedding + LSTM
    + **Task 4** (Cell 32-39): Embedding học từ đầu + LSTM
    + **Task 5** (Cell 40-41): Đánh giá và so sánh tổng thể

Mỗi task được thực hiện tuần tự và kết quả được hiển thị ngay dưới từng cell code.

## Thông tin cấu hình và siêu tham số

### 1. TF-IDF + Logistic Regression
- TF-IDF parameters: max_features=10000, ngram_range=(1,2), lowercase=True
- LogisticRegression: max_iter=1000, random_state=42, multi_class='ovr'
- Training strategy: Fit trực tiếp trên sparse TF-IDF matrix

### 2. Word2Vec + Dense Neural Network  
- Word2Vec: vector_size=200, window=5, min_count=1, workers=4
- Neural Network: 3 hidden layers [128, 64, 32], activation='relu'
- Dropout: 0.3 sau mỗi hidden layer
- Optimizer: Adam, learning_rate=0.001
- Epochs: 50 với EarlyStopping patience=10

### 3. Pre-trained Word2Vec + LSTM
- Pre-trained embedding: Google News Word2Vec (300D), frozen=True  
- LSTM architecture: 128 units, dropout=0.3, recurrent_dropout=0.3
- Dense layers: 64 units → num_classes
- Optimizer: Adam, learning_rate=0.001
- Batch size: 256, epochs=30

### 4. Embedding từ đầu + LSTM
- Embedding layer: 200 dimensions, trainable=True
- LSTM: 128 units với Bidirectional wrapper
- Regularization: BatchNormalization, Dropout(0.5)
- Dense: 64 units, activation='relu' 
- Training: Adam optimizer, batch_size=256, epochs=50

## Bảng so sánh kết quả định lượng

| Phương pháp              |   Accuracy    |   Loss   |  Macro F1-Score   |  Weighted F1-Score |
|------------------------------|---------------|----------|-------------------|---------------------|
| TF-IDF + LogReg          | 0.8589        | 0.9800   | 0.8251            | 0.8567              |
| Word2Vec + Dense NN      | 0.5600        | N/A      | 0.5100            | 0.5400              |
| Pre-trained W2V + LSTM   | 0.2730        | 2.6828   | 0.2300            | 0.2400              |
| Embedding từ đầu + LSTM  | 0.8078        | 0.7899   | 0.7900            | 0.8100              |


### Nhận xét về bảng kết quả:

- TF-IDF + LogReg: Baseline mạnh, nhanh, hiệu quả cao - đạt performance tốt nhất với độ phức tạp thấp
- Word2Vec + Dense NN: Vector trung bình làm mất thông tin sequence, performance trung bình  
- Pre-trained W2V + LSTM: Embedding frozen không phù hợp với domain, performance cực kì thấp
- Embedding từ đầu + LSTM: Embedding trainable học được representation tốt, cân bằng giữa complexity và performance


## Phân tích định tính với ví dụ điển hình thực tế

### Ví dụ 1: Câu có phụ thuộc xa - "Can you please set an alarm for me tomorrow morning at 6 AM"

**Kết quả phân loại:**
- **TF-IDF + LogReg**: alarm_set (prob: 0.8982) - Correct
- **Word2Vec + Dense**: alarm_set (prob: 0.7295) - Correct 
- **Pre-trained LSTM**: alarm_set (prob: 0.8971) - Correct
- **Scratch LSTM**: alarm_set (prob: 0.9691) - Correct

**Phân tích**: Tất cả các model đều dự đoán đúng do từ khóa "alarm" và "set" rõ ràng. TF-IDF thành công nhờ n-gram capture được "set alarm", trong khi LSTM models có khả năng liên kết "set alarm" ở đầu với "6 AM" ở cuối câu qua hidden states. Kết quả cho thấy với câu có cấu trúc rõ ràng, cả phương pháp đơn giản và phức tạp đều hiệu quả.

### Ví dụ 2: Câu phủ định phức tạp - "Don't play music right now"

**Kết quả phân loại:**
- **TF-IDF + LogReg**: play_music (prob: 0.6405) - Wrong
- **Word2Vec + Dense**: play_music (prob: 0.4385) - Wrong
- **Pre-trained LSTM**: news_query (prob: 0.0712) - Wrong
- **Scratch LSTM**: play_music (prob: 0.9348) - Wrong

**Intent đúng**: general_commandstop

**Phân tích**: Tất cả các mô hình đều thất bại trong việc xử lý phủ định. Với TF-IDF, các từ “play” và “music” có trọng số cao, khiến từ “Don’t” không đủ ảnh hưởng để đảo ngược ý nghĩa. Trong Word2Vec, phương pháp trung bình vector làm loãng tín hiệu phủ định. Các mô hình LSTM dù có khả năng xử lý tuần tự nhưng  vẫn chưa học được mẫu câu phủ định trong dữ liệu huấn luyện. Đáng chú ý là Scratch LSTM thể hiện mức độ tự tin rất cao (93.48%) dù hoàn toàn sai, cho thấy hiện tượng overfitting nguy hiểm.

### Ví dụ 3: Câu đơn giản - "Weather today"

**Kết quả phân loại:**
- **TF-IDF + LogReg**: weather_query (prob: 0.9728) - Correct
- **Word2Vec + Dense**: weather_query (prob: 0.6296) - Correct
- **Pre-trained LSTM**: news_query (prob: 0.0982) - Wrong  
- **Scratch LSTM**: weather_query (prob: 0.9954) - Correct

**Phân tích**: TF-IDF thể hiện vượt trội với độ chính xác rất cao, nhờ từ “weather” là đặc trưng mạnh cho lớp weather_query. Word2Vec vẫn đúng, nhưng độ tin cậy thấp hơn do từ “today” làm giảm sức mạnh tín hiệu. Pre-trained LSTM thất bại do embeddings từ Google News không được huấn luyện cho tác vụ phân loại ý định. Trường hợp này cho thấy những mô hình đơn giản như TF-IDF có thể hoạt động rất tốt với câu ngắn, rõ ràng.

### Ví dụ 4: Câu ngoài domain - "find a flight from new york to london"

**Kết quả phân loại:**
- **TF-IDF + LogReg**: general_negate (prob: 0.0784) - Wrong
- **Word2Vec + Dense**: lists_createoradd (prob: 0.1531) - Wrong
- **Pre-trained LSTM**: email_addcontact (prob: 0.1247) - Wrong
- **Scratch LSTM**: social_post (prob: 0.8672) - Wrong

**Intent đúng**: flight_search (không có trong training set)

**Phân tích**: Đây là ví dụ điển hình của vấn đề chuyển miền (domain shift). Các mô hình TF-IDF và Word2Vec không tìm thấy đặc trưng mạnh, nên dự đoán với độ tin cậy thấp – điều này phản ánh sự không chắc chắn hợp lý. Pre-trained LSTM cũng cho thấy tín hiệu yếu (~12%). Tuy nhiên, Scratch LSTM lại thể hiện độ tự tin cao ngược với độ chính xác (86.72%), tạo ra ảo giác tin tưởng sai lệch. Trong môi trường sản xuất, cần có cơ chế threshold hoặc fallback để từ chối những dự đoán thiếu chắc chắn.

### Ví dụ 5: Câu reminder phức tạp - "can you remind me to not call my mom"

**Kết quả phân loại:**
- **TF-IDF + LogReg**: calendar_set (prob: 0.3335) - Wrong
- **Word2Vec + Dense**: calendar_set (prob: 0.3210) - Wrong 
- **Pre-trained LSTM**: social_post (prob: 0.0986) - Wrong
- **Scratch LSTM**: calendar_set (prob: 0.9835) - Wrong

**Intent đúng**: reminder_create (không có trong training set)

**Phân tích**: TF-IDF và Word2Vec đưa ra dự đoán sai nhưng hợp lý – "remind" có liên kết ngữ nghĩa với hành động lên lịch, dẫn đến lựa chọn calendar_set với độ tin cậy trung bình (~33%). Pre-trained LSTM tỏ ra không chắc chắn, với dự đoán ngẫu nhiên và độ tin cậy thấp (9.86%). Ngược lại, Scratch LSTM một lần nữa overconfident khi dự đoán sai với xác suất cực cao (98.35%) – đây là mối nguy lớn nếu mô hình được triển khai thực tế mà không có kiểm soát.

## Nhận xét tổng quan:

**TF-IDF + Logistic Regression** thể hiện reliability cao nhất: strong performance với keyword-based queries, reasonable uncertainty với unfamiliar input. N-gram features và linear decision boundary tạo interpretable và predictable behavior.

**Word2Vec + Dense NN** cho moderate performance. Averaging operation mất spatial information nhưng vẫn preserve semantic similarity. Không có clear advantage so với TF-IDF baseline.

**Pre-trained Word2Vec + LSTM** nghiêm trọng underperform do domain mismatch. Google News embeddings không optimize cho conversational intent classification. Frozen weights không adapt được với task-specific patterns.

**Embedding từ đầu + LSTM** có accuracy cao nhưng dangerous overconfidence. Model học được training patterns tốt nhưng không generalize safely cho out-of-distribution input. High-confidence wrong predictions rất nguy hiểm trong production systems.

## Nhận xét các phương pháp

### 1. TF-IDF + Logistic Regression
**Ưu điểm:**
- Đơn giản, dễ implement và debug
- Training nhanh, ít resource
- Hiệu quả tốt cho text classification cơ bản

**Nhược điểm:**
- Không nắm bắt được ngữ nghĩa sâu
- Bag-of-words mất thông tin về thứ tự từ

### 2. Word2Vec Averaging + Dense NN
**Ưu điểm:**
- Tận dụng semantic information từ Word2Vec
- Mô hình compact, training tương đối nhanh
- Có thể fine-tune qua các layer Dense

**Nhược điểm:**
- Averaging làm mất thông tin về vị trí và thứ tự từ
- Word2Vec training trên corpus nhỏ có thể không tối ưu

### 3. Pre-trained Word2Vec + LSTM
**Ưu điểm:**
- LSTM nắm bắt được sequential information
- Sử dụng knowledge từ pre-trained embeddings
- Ổn định khi embedding đóng băng

**Nhược điểm:**
- Pre-trained embedding có thể không fit với domain cụ thể
- LSTM phức tạp hơn, cần nhiều data để học tốt

### 4. Embedding học từ đầu + LSTM
**Ưu điểm:**
- Embedding được tối ưu cho task cụ thể
- Linh hoạt nhất, có thể học representation phù hợp

**Nhược điểm:**
- Cần nhiều dữ liệu để học embedding tốt
- Training lâu hơn, dễ overfit với dataset nhỏ

## Challenges and Solutions

- Overfitting với dataset nhỏ: Model phức tạp dễ overfit
-> Solution: Dropout, BatchNormalization, L2 regularization, EarlyStopping với patience=10

- Stopwords có thể ảnh hưởng đến performance
-> Solution: Thử nghiệm cả có và không có stopword removal trong Word2Vec averaging (performance tăng)

- Gặp lỗi hiển thị khi setup TensorBoard callbacks để track loss/accuracy curves
-> Sử dụng matplotlib

## Kết luận và nhận xét chung

- Qua thực nghiệm, Word2Vec averaging + Dense NN cho kết quả tốt nhất với balance giữa performance và complexity. TF-IDF baseline vẫn rất competitive và nên được xem xét cho production systems cần tốc độ.

- LSTM hoạt động hiệu quả với câu phức tạp, long dependencies, cấu trúc ngữ pháp. TF-IDF phân loại tốt với câu đơn giản, từ khóa rõ ràng. Word2Vec averaging yếu nhất do mất thông tin structural.

- LSTM models mang lại improvement nhẹ nhưng đáng kể, đặc biệt với các intent có cấu trúc ngữ pháp phức tạp. Tuy nhiên, với dataset HWU tương đối đơn giản, sự khác biệt không quá lớn.

## Cite References

- Dataset: HWU64 - A dataset for intent classification and slot filling
- Gensim Word2Vec: Mikolov et al., "Distributed Representations of Words and Phrases"  
- TensorFlow/Keras: Framework cho deep learning models
- Scikit-learn: Machine learning library cho preprocessing và metrics
- NLTK: Natural Language Toolkit cho stopwords processing
- Tensorboard: track loss/accuracy curves
- Tools: Copilot