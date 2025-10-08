import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root = os.path.dirname(current_dir)
# sys.path.append(root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.dataset_loaders import load_raw_text_data
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer    
dataset_path = "UD_English-EWT/en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)
sample_text = raw_text[:137] 
print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text}...")
simple_tokenizer = SimpleTokenizer()
simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output: {simple_tokens}")
regex_tokenizer = RegexTokenizer()
regex_tokens = regex_tokenizer.tokenize(sample_text)
print(f"RegexTokenizer Output: {regex_tokens}")

print("\n--- Tokenizing Simple text---")
simple_text = "Đi Trung Quốc thổi nến sinh nhật thoaiii! Ai là Support đi Worlds 6 năm liên tiếp từ khi debut đến giờ nào? Là ai "
print(f"Original Simple Text: {simple_text}")   
simple_tokens = simple_tokenizer.tokenize(simple_text)
print(f"SimpleTokenizer Output: {simple_tokens}")
print(f"RegexTokenizer Output: {regex_tokens}")
