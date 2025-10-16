from typing import Optional, Sequence

import numpy as np

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.interfaces import Tokenizer
from src.representations.word_embedder import WordEmbedder


class DocumentEmbedder:
    """Compute a document embedding as the element-wise mean of known word vectors.

    Behavior:
    - Tokenize the document using a Tokenizer (default: RegexTokenizer).
    - For each token, attempt to retrieve its vector from the WordEmbedder.
    - Ignore out-of-vocabulary (OOV) tokens.
    - If no known tokens are present, return a zero vector of the correct dimension.
    - Otherwise return the element-wise mean of all retrieved word vectors.

    The constructor accepts either an already-loaded ``WordEmbedder`` instance or
    a gensim model name (str), in which case a ``WordEmbedder`` is created.
    """

    def __init__(self, embedder: Optional[object] = None, tokenizer: Optional[Tokenizer] = None):
        # embedder may be a WordEmbedder instance or a gensim model name (str)
        if isinstance(embedder, str):
            self.embedder = WordEmbedder(embedder)
        elif isinstance(embedder, WordEmbedder):
            self.embedder = embedder
        elif embedder is None:
            raise ValueError("embedder must be a WordEmbedder instance or a gensim model name string")
        else:
            # try duck-typing: assume it behaves like WordEmbedder
            self.embedder = embedder

        self.tokenizer: Tokenizer = tokenizer if tokenizer is not None else RegexTokenizer()

        # determine embedding dimensionality
        try:
            self.dim = int(self.embedder.vector_size())
        except Exception:
            # fallback to None; will handle at embed time
            self.dim = None

    def embed(self, document: str) -> np.ndarray:
        """Return a dense numpy vector for the input document.

        Args:
            document: input text

        Returns:
            numpy.ndarray of shape (dim,) where dim is the embedding size.
        """
        tokens = self.tokenizer.tokenize(document or "")

        vecs = []
        for t in tokens:
            # generator may raise or support membership; try safe access
            try:
                if t in self.embedder:
                    v = self.embedder.get_vector(t)
                    vecs.append(np.asarray(v, dtype=float))
            except Exception:
                # ignore tokens we cannot retrieve
                continue

        if len(vecs) == 0:
            # return zero vector of correct dimension
            if self.dim is None:
                # try to recover dimension from embedder one more time
                try:
                    self.dim = int(self.embedder.vector_size())
                except Exception:
                    raise RuntimeError("Unable to determine embedding dimension")
            return np.zeros(self.dim, dtype=float)

        # compute element-wise mean
        stacked = np.vstack(vecs)
        return np.mean(stacked, axis=0)

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        """Embed multiple documents; returns array of shape (n_documents, dim)."""
        outputs = [self.embed(doc) for doc in documents]
        return np.vstack(outputs)

