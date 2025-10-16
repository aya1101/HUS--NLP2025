from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectoriser import CountVectorizer

def testSimpleCountVectorizer():
    print("\n--- Testing Simple CountVectorizer ---")
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    X = vectorizer.fit_transform(corpus)

    print("Learned vocabulary:")
    print(vectorizer.vocabulary_)
    print("\nDocument-term matrix:")
    for row in X:
        print(row)

def testFullCountVectorizer():
    print("\n--- Testing CountVectorizer with Sample Text from UD_English-EWT ---")
    from src.core.dataset_loaders import load_raw_text_data
    dataset_path = "UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    sample_texts = raw_text.split("\n\n")[:88]  # First 3 sentences

    tokenizer = RegexTokenizer()
    vectorizer = CountVectorizer(tokenizer)

    X = vectorizer.fit_transform(sample_texts)

    print("Learned vocabulary from sample texts:")
    print(vectorizer.vocabulary_)
    print("\nDocument-term matrix for sample texts:")
    for row in X:
        print(row)

if __name__ == "__main__":
    testSimpleCountVectorizer()
    testFullCountVectorizer()
