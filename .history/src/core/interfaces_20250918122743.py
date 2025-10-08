from abc import ABC, abstractmethod

class Tokenizer(ABC):
	@abstractmethod
	def tokenize(self, text: str) -> list[str]:
		"""Tokenize the input text into a list of strings."""
		pass
	
class Vectorizer(ABC):
	@abstractmethod
	#def fit(self, corpus: list[str]):
        
	def transform(self, documents:list[str])-> list[list[int]]:
		pass
		
    #def fit_transform(self, corpus: list[str]) -> list[list[int]]:
	#	pass

