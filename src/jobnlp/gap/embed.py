"""multilingual-e5-small sentence embeddings (loaded lazily, cached per string)."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"


class E5Embedder:
    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if new:
            vecs = self.model.encode(
                ["query: " + t for t in new], normalize_embeddings=True, batch_size=64
            )
            for t, v in zip(new, vecs, strict=True):
                self._cache[t] = np.asarray(v, dtype=np.float32)
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, 384))

    def similarity(self, a: list[str], b: list[str]) -> list[list[float]]:
        if not a or not b:
            return [[] for _ in a]
        return (self.encode(a) @ self.encode(b).T).tolist()


@lru_cache(maxsize=1)
def get_embedder() -> E5Embedder:
    return E5Embedder()
