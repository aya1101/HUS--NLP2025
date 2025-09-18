# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root = os.path.dirname(current_dir)
# sys.path.append(root)
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.dataset_loaders import load_raw_text_data
from src.preprocessing.simple_tokenizer import SimpleTokenizer
from src.preprocessing.regex_tokenizer import RegexTokenizer
dataset_path = "UD_English-EWT/en_ewt-ud-train.txt"
raw_text = load_raw_text_data(dataset_path)
sample_text = raw_text[:137] 
print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
print(f"Original Sample: {sample_text[:100]}...")
simple_tokenizer = SimpleTokenizer()
simple_tokens = simple_tokenizer.tokenize(sample_text)
print(f"SimpleTokenizer Output (first 20 tokens): {simple_tokens[:20]}")
regex_tokenizer = RegexTokenizer()
regex_tokens = regex_tokenizer.tokenize(sample_text)
print(f"RegexTokenizer Output (first 20 tokens): {regex_tokens[:20]}")