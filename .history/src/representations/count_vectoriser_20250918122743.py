from src.core.interfaces import Tokenizer, Vectorizer
class CountVectorizer(Vectorizer):
    vocabbulary_ = {}
    #def __init__(self, tokenizer: Tokenizer):
        