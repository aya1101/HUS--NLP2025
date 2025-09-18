from src.core.interfaces import Tokenizer
class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        import re
        # Extract words and punctuation as separate tokens
        return re.findall(r'\w+|[^\w\s]', text.lower())