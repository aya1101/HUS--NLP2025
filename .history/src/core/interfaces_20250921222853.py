from abc import ABC, abstractmethod

class Tokenizer(ABC):
	@abstractmethod
	def tokenize(self, text: str) -> list[str]:
		"""Tokenize the input text into a list of strings."""
		pass
	
class Vectorizer(ABC):

	@abstractmethod
	def fit(self, corpus: list[str]):
		"""Learns the vocabulary from a list of documents (corpus)."""
		pass

	@abstractmethod
	def transform(self, documents: list[str]) -> list[list[int]]:
		"""Transforms a list of documents into a list of count vectors based on the learned vocabulary."""
		pass

	@abstractmethod
	def fit_transform(self, corpus: list[str]) -> list[list[int]]:
		"""Performs fit and then transform on the same data."""
		pass

