from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectoriser import CountVectorizer

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
