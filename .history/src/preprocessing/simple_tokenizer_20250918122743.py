from src.core.interfaces import Tokenizer
class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()