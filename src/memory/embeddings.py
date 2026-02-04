"""Embedding generation for memory vectors"""

from typing import List, Optional
import numpy as np

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


class EmbeddingModel:
    """
    Wrapper for embedding model (all-MiniLM-L6-v2).

    Uses sentence-transformers for lightweight, fast embeddings.
    384-dimensional vectors that capture semantic meaning.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

    def __init__(self, device: str = "cpu"):
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(self.MODEL_NAME, device=device)
        self.device = device

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for single text"""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)


class MockEmbeddingModel:
    """Mock embedding model for testing without GPU/dependencies"""

    EMBEDDING_DIM = 384

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.rng = np.random.default_rng(42)

    def embed(self, text: str) -> np.ndarray:
        """Generate random embedding (deterministic based on text hash)"""
        seed = hash(text) % (2**32)
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(self.EMBEDDING_DIM)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        return np.array([self.embed(t) for t in texts], dtype=np.float32)


def create_embedding_model(use_mock: bool = False, device: str = "cpu") -> EmbeddingModel:
    """Factory to create embedding model"""
    if use_mock or not EMBEDDINGS_AVAILABLE:
        return MockEmbeddingModel(device=device)
    return EmbeddingModel(device=device)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query and multiple vectors"""
    # Normalize query
    query_norm = query / np.linalg.norm(query)

    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    vectors_norm = vectors / norms

    # Compute similarities
    similarities = np.dot(vectors_norm, query_norm)
    return similarities
