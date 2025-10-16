"""Simple wrapper for loading word embedding models via gensim downloader.

This module provides a minimal WordEmbedder class. The constructor accepts a
model name (for example 'glove-wiki-gigaword-50') and loads it using
gensim.downloader.load so you can access vectors as `embedder.get_vector(word)`.
"""

from typing import Any

import gensim.downloader as api


class WordEmbedder:
        def __init__(self, model_name: str):
                """Load a pre-trained embedding model by name.

                Args:
                        model_name: a model id known to gensim.downloader (e.g.
                                'glove-wiki-gigaword-50'). The model will be downloaded and
                                cached by gensim if not available locally.
                """
                self.model_name = model_name
                # gensim will download the model if it's not already cached.
                self.model = api.load(model_name)

        def __contains__(self, word: str) -> bool:
                try:
                        return word in self.model
                except Exception:
                        return False

        def get_vector(self, word: str) -> Any:
                """Return the vector for a given word.

                Raises a KeyError if the word is not in the model's vocabulary.
                """
                return self.model[word]

        def vector_size(self) -> int:
                """Return embedding size (number of dimensions)."""
                try:
                        return self.model.vector_size
                except AttributeError:
                        # Some older KeyedVectors expose 'vector_size' differently
                        return getattr(self.model, 'vectors.shape', (None, None))[1]

