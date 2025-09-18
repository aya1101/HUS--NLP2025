from src.core.interfaces import Tokenizer
class RegexTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        import re
        return re.findall(r'\b\w+\b', text.lower())