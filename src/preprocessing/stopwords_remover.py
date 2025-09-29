def remove_stopwords(text: str, stopwords: set) -> str:
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    return ' '.join(filtered_tokens)