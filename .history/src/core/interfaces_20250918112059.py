from abc import ABC, abstractmethod

class Tokenizer(ABC):
	@abstractmethod
	def tokenize(self, text: str) -> list[str]:
		"""Tokenize the input text into a list of strings."""
		pass
