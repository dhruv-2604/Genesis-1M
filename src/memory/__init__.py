# Memory module for agent memory streams (Phase 3)

from .schema import Memory, MemoryType, Relationship, create_memory_id
from .embeddings import (
    EmbeddingModel,
    MockEmbeddingModel,
    create_embedding_model,
    cosine_similarity,
    EMBEDDINGS_AVAILABLE,
)
from .store import MemoryStore, LANCEDB_AVAILABLE
from .manager import MemoryManager

__all__ = [
    "Memory",
    "MemoryType",
    "Relationship",
    "create_memory_id",
    "EmbeddingModel",
    "MockEmbeddingModel",
    "create_embedding_model",
    "cosine_similarity",
    "EMBEDDINGS_AVAILABLE",
    "MemoryStore",
    "LANCEDB_AVAILABLE",
    "MemoryManager",
]
