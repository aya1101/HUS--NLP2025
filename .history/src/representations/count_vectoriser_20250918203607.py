from src.core.interfaces import Tokenizer, Vectorizer

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: dict[str, int] = {}
        