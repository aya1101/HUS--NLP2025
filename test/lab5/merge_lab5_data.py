import os
import pandas as pd
import re

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lab5'))
output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'lab5', 'sent_all_cleaned.csv'))

train_file = os.path.join(data_dir, 'sent_train.csv')
valid_file = os.path.join(data_dir, 'sent_valid.csv')

df_train = pd.read_csv(train_file)
df_valid = pd.read_csv(valid_file)

print(f"Train data shape: {df_train.shape}")
print(f"Valid data shape: {df_valid.shape}")

df_combined = pd.concat([df_train, df_valid], ignore_index=True)

print(f"\nCombined data shape before preprocessing: {df_combined.shape}")

# Preprocessing
print("\n=== TIỀN XỬ LÝ DỮ LIỆU ===")

# 1. Loại bỏ null/NaN
print(f"Null values before: {df_combined.isnull().sum().sum()}")
df_combined = df_combined.dropna()
print(f"Data shape after removing nulls: {df_combined.shape}")

# 2. Loại bỏ duplicate
duplicates_before = df_combined.shape[0]
df_combined = df_combined.drop_duplicates(subset=['text'], keep='first')
duplicates_removed = duplicates_before - df_combined.shape[0]
print(f"Removed {duplicates_removed} duplicate texts")
print(f"Data shape after removing duplicates: {df_combined.shape}")

# 3. Làm sạch text
def clean_text(text):
    if not isinstance(text, str):
        return text
    # Loại bỏ khoảng trắng dư thừa
    text = ' '.join(text.split())
    # Loại bỏ URL
    text = re.sub(r'http\S+|www\S+', '', text)
    # Loại bỏ ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\s]', ' ', text)
    # Loại bỏ khoảng trắng dư thừa lần cuối
    text = ' '.join(text.split())
    return text

print("Cleaning text...")
df_combined['text'] = df_combined['text'].apply(clean_text)

# 4. Loại bỏ text trống
df_combined = df_combined[df_combined['text'].str.len() > 0]
print(f"Data shape after removing empty texts: {df_combined.shape}")

# 5. Xác thực label
valid_labels = df_combined['label'].unique()
print(f"Valid labels: {sorted(valid_labels)}")

# 6. Reset index
df_combined = df_combined.reset_index(drop=True)

print(f"\n=== KẾT QUẢ CUỐI CÙNG ===")
print(f"Final data shape: {df_combined.shape}")
print(f"\nPhân bố nhãn:")
print(df_combined['label'].value_counts().sort_index())

# Lưu file
df_combined.to_csv(output_file, index=False)
print(f"\nDữ liệu đã được tiền xử lý và lưu tại: {output_file}")
