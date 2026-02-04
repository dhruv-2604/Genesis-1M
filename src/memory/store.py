"""Memory Store using LanceDB for vector storage"""

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import json

from .schema import Memory, MemoryType, Relationship, create_memory_id
from .embeddings import create_embedding_model, batch_cosine_similarity, EmbeddingModel

# Try to import LanceDB
try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


class MemoryStore:
    """
    Vector database for agent memories using LanceDB.

    Supports:
    - Fast semantic search via embeddings
    - Filtering by agent, type, time range
    - Importance-based pruning
    - Relationship tracking
    """

    MAX_MEMORIES_PER_AGENT = 50  # Prune beyond this

    def __init__(
        self,
        db_path: str = "./memory_db",
        embedding_model: Optional[EmbeddingModel] = None,
        use_mock: bool = False,
    ):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model or create_embedding_model(use_mock=use_mock)
        self.use_mock = use_mock or not LANCEDB_AVAILABLE

        if not self.use_mock:
            self.db = lancedb.connect(str(self.db_path))
            self._init_tables()
        else:
            # In-memory mock store
            self.memories: Dict[str, Memory] = {}
            self.agent_memories: Dict[int, List[str]] = {}  # agent_id -> memory_ids

        # Relationship cache
        self.relationships: Dict[Tuple[int, int], Relationship] = {}

        # Stats
        self.total_stored = 0
        self.total_queries = 0

    def _init_tables(self):
        """Initialize LanceDB tables"""
        # Define schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("agent_id", pa.int64()),
            pa.field("tick", pa.int64()),
            pa.field("memory_type", pa.string()),
            pa.field("summary", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), 384)),
            pa.field("emotional_valence", pa.float32()),
            pa.field("importance", pa.float32()),
            pa.field("other_agent_id", pa.int64()),
            pa.field("location_x", pa.float32()),
            pa.field("location_y", pa.float32()),
            pa.field("inherited_from", pa.int64()),
            pa.field("generation", pa.int32()),
            pa.field("access_count", pa.int32()),
            pa.field("last_accessed", pa.int64()),
        ])

        # Create or open table
        if "memories" not in self.db.table_names():
            self.table = self.db.create_table("memories", schema=schema)
        else:
            self.table = self.db.open_table("memories")

    def store(self, memory: Memory) -> None:
        """Store a memory with embedding"""
        # Generate embedding if not present
        if memory.embedding is None:
            memory.embedding = self.embedding_model.embed(memory.summary)

        if self.use_mock:
            self.memories[memory.id] = memory
            if memory.agent_id not in self.agent_memories:
                self.agent_memories[memory.agent_id] = []
            self.agent_memories[memory.agent_id].append(memory.id)
        else:
            data = memory.to_dict()
            self.table.add([data])

        # Update relationship if involves another agent
        if memory.other_agent_id is not None:
            self._update_relationship(
                memory.agent_id,
                memory.other_agent_id,
                memory.emotional_valence,
                memory.tick
            )

        self.total_stored += 1

        # Check if pruning needed
        self._maybe_prune(memory.agent_id)

    def store_batch(self, memories: List[Memory]) -> None:
        """Store multiple memories"""
        for memory in memories:
            self.store(memory)

    def query_by_agent(
        self,
        agent_id: int,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> List[Memory]:
        """Get recent memories for an agent"""
        self.total_queries += 1

        if self.use_mock:
            memory_ids = self.agent_memories.get(agent_id, [])
            memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]

            if memory_type:
                memories = [m for m in memories if m.memory_type == memory_type]

            # Sort by tick (most recent first)
            memories.sort(key=lambda m: m.tick, reverse=True)
            return memories[:limit]
        else:
            query = f"agent_id = {agent_id}"
            if memory_type:
                query += f" AND memory_type = '{memory_type.name}'"

            results = self.table.search().where(query).limit(limit).to_list()
            return [Memory.from_dict(r) for r in results]

    def query_semantic(
        self,
        agent_id: int,
        query_text: str,
        limit: int = 5,
    ) -> List[Tuple[Memory, float]]:
        """Semantic search for memories matching query"""
        self.total_queries += 1

        query_embedding = self.embedding_model.embed(query_text)

        if self.use_mock:
            memory_ids = self.agent_memories.get(agent_id, [])
            memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]

            if not memories:
                return []

            # Compute similarities
            embeddings = np.array([m.embedding for m in memories if m.embedding is not None])
            if len(embeddings) == 0:
                return []

            similarities = batch_cosine_similarity(query_embedding, embeddings)

            # Sort by similarity
            indices = np.argsort(similarities)[::-1][:limit]
            return [(memories[i], float(similarities[i])) for i in indices]
        else:
            results = (
                self.table.search(query_embedding)
                .where(f"agent_id = {agent_id}")
                .limit(limit)
                .to_list()
            )
            return [(Memory.from_dict(r), r.get('_distance', 0)) for r in results]

    def query_about_agent(
        self,
        agent_id: int,
        about_agent_id: int,
        limit: int = 5,
    ) -> List[Memory]:
        """Get memories about a specific other agent"""
        self.total_queries += 1

        if self.use_mock:
            memory_ids = self.agent_memories.get(agent_id, [])
            memories = [
                self.memories[mid]
                for mid in memory_ids
                if mid in self.memories and self.memories[mid].other_agent_id == about_agent_id
            ]
            memories.sort(key=lambda m: m.tick, reverse=True)
            return memories[:limit]
        else:
            results = (
                self.table.search()
                .where(f"agent_id = {agent_id} AND other_agent_id = {about_agent_id}")
                .limit(limit)
                .to_list()
            )
            return [Memory.from_dict(r) for r in results]

    def get_relationship(self, agent_id: int, other_agent_id: int) -> Optional[Relationship]:
        """Get relationship between two agents"""
        key = (min(agent_id, other_agent_id), max(agent_id, other_agent_id))
        return self.relationships.get(key)

    def _update_relationship(
        self,
        agent_id: int,
        other_agent_id: int,
        valence: float,
        tick: int
    ) -> None:
        """Update relationship based on interaction"""
        key = (min(agent_id, other_agent_id), max(agent_id, other_agent_id))

        if key not in self.relationships:
            self.relationships[key] = Relationship(
                agent_id=key[0],
                other_agent_id=key[1],
            )

        rel = self.relationships[key]
        rel.interaction_count += 1
        rel.total_valence += valence
        rel.last_interaction_tick = tick

    def _maybe_prune(self, agent_id: int) -> None:
        """Prune old/unimportant memories if over limit"""
        if self.use_mock:
            memory_ids = self.agent_memories.get(agent_id, [])
            if len(memory_ids) <= self.MAX_MEMORIES_PER_AGENT:
                return

            # Get all memories
            memories = [(mid, self.memories[mid]) for mid in memory_ids if mid in self.memories]

            # Score = importance * recency_factor
            current_tick = max(m.tick for _, m in memories) if memories else 0

            def retention_score(m: Memory) -> float:
                recency = 1.0 / (1.0 + (current_tick - m.tick) / 1000)
                access_bonus = 0.1 * m.access_count
                return m.importance * recency + access_bonus

            # Sort by retention score
            memories.sort(key=lambda x: retention_score(x[1]), reverse=True)

            # Keep top N
            keep_ids = set(mid for mid, _ in memories[:self.MAX_MEMORIES_PER_AGENT])
            remove_ids = [mid for mid in memory_ids if mid not in keep_ids]

            for mid in remove_ids:
                del self.memories[mid]

            self.agent_memories[agent_id] = list(keep_ids)

    def inherit_memories(
        self,
        parent_id: int,
        child_id: int,
        current_tick: int,
        inherit_fraction: float = 0.3,
    ) -> int:
        """
        Transfer subset of memories from parent to child.
        Returns number of memories inherited.
        """
        parent_memories = self.query_by_agent(parent_id, limit=100)

        if not parent_memories:
            return 0

        # Select most important memories
        parent_memories.sort(key=lambda m: m.importance, reverse=True)
        num_inherit = max(1, int(len(parent_memories) * inherit_fraction))

        inherited = 0
        for memory in parent_memories[:num_inherit]:
            # Create inherited copy
            new_memory = Memory(
                id=create_memory_id(child_id, current_tick, f"inh_{memory.id}"),
                agent_id=child_id,
                tick=current_tick,
                memory_type=MemoryType.INHERITED,
                summary=f"[Story from parent] {memory.summary}",
                embedding=memory.embedding.copy() if memory.embedding is not None else None,
                emotional_valence=memory.emotional_valence * 0.5,  # Diluted
                importance=memory.importance * 0.7,  # Less important than own memories
                other_agent_id=memory.other_agent_id,
                inherited_from=parent_id,
                generation=memory.generation + 1,
            )
            self.store(new_memory)
            inherited += 1

        return inherited

    def delete_agent_memories(self, agent_id: int) -> int:
        """Delete all memories for an agent (when they die)"""
        if self.use_mock:
            memory_ids = self.agent_memories.pop(agent_id, [])
            for mid in memory_ids:
                self.memories.pop(mid, None)
            return len(memory_ids)
        else:
            # LanceDB deletion
            count = self.table.delete(f"agent_id = {agent_id}")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        if self.use_mock:
            num_memories = len(self.memories)
            num_agents = len(self.agent_memories)
        else:
            num_memories = self.table.count_rows()
            num_agents = len(set(r['agent_id'] for r in self.table.to_pandas()['agent_id']))

        return {
            'total_stored': self.total_stored,
            'total_queries': self.total_queries,
            'current_memories': num_memories,
            'agents_with_memories': num_agents,
            'relationships': len(self.relationships),
            'using_lancedb': not self.use_mock,
        }
