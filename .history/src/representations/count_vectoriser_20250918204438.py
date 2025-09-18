from src.core.interfaces import Tokenizer, Vectorizer

class CountVectorizer(Vectorizer):
    def fit(self, corpus: list[str]):
        unique_tokens = set()
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

    def transform(self, documents: list[str]) -> list[list[int]]:
        vectors = []
        vocab_size = len(self.vocabulary_)
        for doc in documents:
            vec = [0] * vocab_size
            tokens = self.tokenizer.tokenize(doc)
            for token in tokens:
                idx = self.vocabulary_.get(token)
                if idx is not None:
                    vec[idx] += 1
            vectors.append(vec)
        return vectors
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary_: dict[str, int] = {}
        