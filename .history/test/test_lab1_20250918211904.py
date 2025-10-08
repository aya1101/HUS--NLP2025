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
simple_text = '''
Slice-of-Life Romcom "Houkago Kitaku Biyori" has won the Next Manga Award 2025 Special Nichirei Award!

ANIME in the works!

Romcom by Matsuda Mai about a cheerful high school girl having exciting "ordinary" daily adventures as a member of the "Going Home" club and her new member, a boy who reluctantly tags along.
'''
print(f"Original Simple Text: {simple_text}")
simple_tokens = simple_tokenizer.tokenize(simple_text)
regex_tokens_vn = regex_tokenizer.tokenize(simple_text)
print(f"SimpleTokenizer Output: {simple_tokens}")
print(f"RegexTokenizer Output: {regex_tokens_vn}")
