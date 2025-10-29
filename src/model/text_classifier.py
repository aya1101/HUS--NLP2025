from typing import List, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TextClassifier:

    def __init__(self, vectorizer) -> None:
        self.vectorizer = vectorizer
        self._model: LogisticRegression | None = None

    def fit(self, texts: List[str], labels: List[int]):
        X = self.vectorizer.fit_transform(texts)
        self._model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        if self._model is None:
            raise RuntimeError("Model chưa được huấn luyện. Gọi fit() trước khi predict().")
        X = self.vectorizer.transform(texts)
        preds = self._model.predict(X)
        return preds.tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
