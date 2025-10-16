from typing import Optional, Sequence

import numpy as np

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.core.interfaces import Tokenizer
from src.representations.word_embedder import WordEmbedder


class DocumentEmbedder:
    """Tạo embedding cho một văn bản bằng trung bình theo phần tử của các vector từ biết.

    Hành vi:
    - Tách văn bản thành token sử dụng một `Tokenizer` (mặc định: `RegexTokenizer`).
    - Với mỗi token, cố gắng lấy vector tương ứng từ `WordEmbedder`.
    - Bỏ qua các token ngoài từ vựng (OOV).
    - Nếu không có token nào được biết đến, trả về vector không (zero vector) có kích thước đúng.
    - Ngược lại trả về trung bình theo phần tử của tất cả các vector từ thu được.

    Constructor chấp nhận một thể hiện `WordEmbedder` đã được tải sẵn hoặc
    một chuỗi tên mô hình gensim (ví dụ: 'glove-wiki-gigaword-50'), trong
    trường hợp đó sẽ tạo một `WordEmbedder` mới từ tên mô hình.
    """

    def __init__(self, embedder: Optional[object] = None, tokenizer: Optional[Tokenizer] = None):
        # `embedder` có thể là một thể hiện WordEmbedder hoặc tên mô hình gensim (str)
        if isinstance(embedder, str):
            self.embedder = WordEmbedder(embedder)
        elif isinstance(embedder, WordEmbedder):
            self.embedder = embedder
        elif embedder is None:
            raise ValueError("embedder must be a WordEmbedder instance or a gensim model name string")
        else:
            # thử duck-typing: giả sử đối tượng có API giống WordEmbedder
            self.embedder = embedder

        self.tokenizer: Tokenizer = tokenizer if tokenizer is not None else RegexTokenizer()

        # Xác định kích thước embedding
        try:
            self.dim = int(self.embedder.vector_size())
        except Exception:
            # nếu không thể xác định được bây giờ thì giữ None; sẽ cố gắng lại khi embed
            self.dim = None

    def embed(self, document: str) -> np.ndarray:
        """Trả về một vector numpy (dense) cho văn bản đầu vào.

        Args:
            document: chuỗi văn bản đầu vào

        Returns:
            numpy.ndarray có hình (dim,) với dim là kích thước embedding.
        """
        tokens = self.tokenizer.tokenize(document or "")

        vecs = []
        for t in tokens:
            # Thử lấy vector cho token. Một số embedder có thể hỗ trợ phép kiểm tra 'in'
            # hoặc ném ngoại lệ khi truy xuất từ ngoài từ vựng; xử lý an toàn ở đây.
            try:
                if t in self.embedder:
                    v = self.embedder.get_vector(t)
                    vecs.append(np.asarray(v, dtype=float))
            except Exception:
                # Bỏ qua token không thể truy xuất (OOV hoặc lỗi khác)
                continue

        if len(vecs) == 0:
            # Nếu không có token nào biết, trả về vector không với kích thước đúng.
            if self.dim is None:
                # Cố gắng lấy lại kích thước embedding từ embedder
                try:
                    self.dim = int(self.embedder.vector_size())
                except Exception:
                    raise RuntimeError("Không thể xác định kích thước embedding")
            return np.zeros(self.dim, dtype=float)

    # Tính trung bình theo phần tử của các vector
        stacked = np.vstack(vecs)
        return np.mean(stacked, axis=0)

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        """Tạo embedding cho nhiều tài liệu; trả về mảng có kích thước (n_documents, dim)."""
        outputs = [self.embed(doc) for doc in documents]
        return np.vstack(outputs)

