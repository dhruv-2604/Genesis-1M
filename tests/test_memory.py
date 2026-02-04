"""Tests for memory system"""

import pytest
import numpy as np
import tempfile
import shutil

from src.memory.schema import Memory, MemoryType, Relationship, create_memory_id
from src.memory.embeddings import MockEmbeddingModel, cosine_similarity, batch_cosine_similarity
from src.memory.store import MemoryStore
from src.memory.manager import MemoryManager


class TestMemorySchema:
    def test_create_memory(self):
        memory = Memory(
            id="1_100_test",
            agent_id=1,
            tick=100,
            memory_type=MemoryType.INTERACTION,
            summary="Met another human",
        )

        assert memory.agent_id == 1
        assert memory.tick == 100
        assert memory.memory_type == MemoryType.INTERACTION

    def test_memory_serialization(self):
        memory = Memory(
            id="1_100_test",
            agent_id=1,
            tick=100,
            memory_type=MemoryType.TRADE,
            summary="Traded berries for flint",
            embedding=np.random.randn(384).astype(np.float32),
            emotional_valence=0.5,
            other_agent_id=2,
        )

        data = memory.to_dict()
        restored = Memory.from_dict(data)

        assert restored.agent_id == memory.agent_id
        assert restored.memory_type == memory.memory_type
        assert restored.summary == memory.summary
        assert restored.other_agent_id == memory.other_agent_id
        assert np.allclose(restored.embedding, memory.embedding)

    def test_create_memory_id(self):
        mid = create_memory_id(42, 1000, "test")
        assert "42" in mid
        assert "1000" in mid
        assert "test" in mid

    def test_relationship_trust_score(self):
        rel = Relationship(agent_id=1, other_agent_id=2)

        # No interactions
        assert rel.trust_score == 0.0

        # Positive interactions
        rel.interaction_count = 5
        rel.total_valence = 2.5

        assert rel.trust_score == 0.5

    def test_relationship_familiarity(self):
        rel = Relationship(agent_id=1, other_agent_id=2)

        # No interactions
        assert rel.familiarity == 0.0

        # Many interactions
        rel.interaction_count = 100
        assert rel.familiarity > 0.9


class TestEmbeddings:
    def test_mock_embedding_deterministic(self):
        model = MockEmbeddingModel()

        emb1 = model.embed("hello world")
        emb2 = model.embed("hello world")

        assert np.allclose(emb1, emb2)

    def test_mock_embedding_different_for_different_text(self):
        model = MockEmbeddingModel()

        emb1 = model.embed("hello world")
        emb2 = model.embed("goodbye world")

        assert not np.allclose(emb1, emb2)

    def test_mock_embedding_shape(self):
        model = MockEmbeddingModel()
        emb = model.embed("test")

        assert emb.shape == (384,)
        assert emb.dtype == np.float32

    def test_batch_embed(self):
        model = MockEmbeddingModel()
        texts = ["hello", "world", "test"]

        embeddings = model.embed_batch(texts)

        assert embeddings.shape == (3, 384)

    def test_cosine_similarity(self):
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([1, 0, 0], dtype=np.float32)
        c = np.array([0, 1, 0], dtype=np.float32)

        assert cosine_similarity(a, b) == pytest.approx(1.0)
        assert cosine_similarity(a, c) == pytest.approx(0.0)

    def test_batch_cosine_similarity(self):
        query = np.array([1, 0, 0], dtype=np.float32)
        vectors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0.7, 0.7, 0],
        ], dtype=np.float32)

        sims = batch_cosine_similarity(query, vectors)

        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)
        assert sims[2] > 0


class TestMemoryStore:
    @pytest.fixture
    def store(self):
        return MemoryStore(use_mock=True)

    def test_store_and_retrieve(self, store):
        memory = Memory(
            id="1_100_test",
            agent_id=1,
            tick=100,
            memory_type=MemoryType.INTERACTION,
            summary="Met Human-2",
            other_agent_id=2,
        )

        store.store(memory)
        results = store.query_by_agent(1)

        assert len(results) == 1
        assert results[0].summary == "Met Human-2"

    def test_query_by_type(self, store):
        store.store(Memory(
            id="1_100_int", agent_id=1, tick=100,
            memory_type=MemoryType.INTERACTION, summary="Met someone"
        ))
        store.store(Memory(
            id="1_101_trade", agent_id=1, tick=101,
            memory_type=MemoryType.TRADE, summary="Traded stuff"
        ))

        interactions = store.query_by_agent(1, memory_type=MemoryType.INTERACTION)
        trades = store.query_by_agent(1, memory_type=MemoryType.TRADE)

        assert len(interactions) == 1
        assert len(trades) == 1
        assert interactions[0].memory_type == MemoryType.INTERACTION

    def test_semantic_search(self, store):
        store.store(Memory(
            id="1_100_food", agent_id=1, tick=100,
            memory_type=MemoryType.DISCOVERY, summary="Found berries near river"
        ))
        store.store(Memory(
            id="1_101_fight", agent_id=1, tick=101,
            memory_type=MemoryType.CONFLICT, summary="Fought with stranger"
        ))

        # Search for food-related memories
        results = store.query_semantic(1, "food berries eating", limit=1)

        assert len(results) >= 1
        # First result should be more relevant to food

    def test_query_about_agent(self, store):
        store.store(Memory(
            id="1_100_a", agent_id=1, tick=100,
            memory_type=MemoryType.INTERACTION, summary="Met Human-2",
            other_agent_id=2
        ))
        store.store(Memory(
            id="1_101_b", agent_id=1, tick=101,
            memory_type=MemoryType.INTERACTION, summary="Met Human-3",
            other_agent_id=3
        ))

        about_2 = store.query_about_agent(1, 2)
        about_3 = store.query_about_agent(1, 3)

        assert len(about_2) == 1
        assert len(about_3) == 1
        assert about_2[0].other_agent_id == 2

    def test_relationship_tracking(self, store):
        # Positive interaction
        store.store(Memory(
            id="1_100_pos", agent_id=1, tick=100,
            memory_type=MemoryType.TRADE, summary="Good trade",
            emotional_valence=0.5, other_agent_id=2
        ))

        rel = store.get_relationship(1, 2)

        assert rel is not None
        assert rel.interaction_count == 1
        assert rel.trust_score == pytest.approx(0.5)

    def test_pruning(self, store):
        store.MAX_MEMORIES_PER_AGENT = 5

        # Store more than max
        for i in range(10):
            store.store(Memory(
                id=f"1_{i}_test", agent_id=1, tick=i,
                memory_type=MemoryType.OBSERVATION, summary=f"Memory {i}",
                importance=0.5 + (i * 0.05)  # Later = more important
            ))

        # Should be pruned to max
        results = store.query_by_agent(1, limit=100)
        assert len(results) <= store.MAX_MEMORIES_PER_AGENT

    def test_inherit_memories(self, store):
        # Parent has memories
        for i in range(5):
            store.store(Memory(
                id=f"1_{i}_test", agent_id=1, tick=i,
                memory_type=MemoryType.DISCOVERY, summary=f"Found something {i}",
                importance=0.8
            ))

        # Child inherits
        inherited = store.inherit_memories(parent_id=1, child_id=2, current_tick=100)

        assert inherited > 0

        # Check child has inherited memories
        child_memories = store.query_by_agent(2)
        assert len(child_memories) > 0
        assert any(m.memory_type == MemoryType.INHERITED for m in child_memories)


class TestMemoryManager:
    @pytest.fixture
    def manager(self):
        return MemoryManager(use_mock=True)

    def test_record_interaction(self, manager):
        memory = manager.record_interaction(
            agent_id=1,
            other_agent_id=2,
            tick=100,
            action="shared food",
            outcome="They seemed grateful",
            valence=0.5,
        )

        assert memory.agent_id == 1
        assert memory.other_agent_id == 2
        assert "Human-2" in memory.summary

    def test_record_trade(self, manager):
        memory = manager.record_trade(
            agent_id=1,
            other_agent_id=2,
            tick=100,
            gave={"berries": 3},
            received={"flint": 1},
        )

        assert "berries" in memory.summary
        assert "flint" in memory.summary
        assert memory.memory_type == MemoryType.TRADE

    def test_record_conflict(self, manager):
        memory = manager.record_conflict(
            agent_id=1,
            other_agent_id=2,
            tick=100,
            won=True,
        )

        assert "won" in memory.summary
        assert memory.emotional_valence > 0

    def test_flush_pending(self, manager):
        manager.record_interaction(1, 2, 100, "met", "ok", 0.0)
        manager.record_interaction(1, 3, 101, "met", "ok", 0.0)

        assert len(manager.pending_memories) == 2

        flushed = manager.flush_pending()

        assert flushed == 2
        assert len(manager.pending_memories) == 0

    def test_recall_for_prompt(self, manager):
        # Store some memories
        manager.record_interaction(1, 2, 100, "traded", "good", 0.5)
        manager.record_conflict(1, 3, 101, won=False)
        manager.flush_pending()

        # Recall
        summaries = manager.recall_for_prompt(
            agent_id=1,
            current_tick=200,
            nearby_agent_ids=[2, 3],
        )

        assert len(summaries) > 0
        assert all(isinstance(s, str) for s in summaries)

    def test_get_relationship_summary(self, manager):
        # Build relationship through interactions
        for i in range(5):
            manager.record_interaction(1, 2, i, "helped", "grateful", 0.4)
        manager.flush_pending()

        summary = manager.get_relationship_summary(1, 2)

        assert summary is not None
        assert "friendly" in summary or "5 interactions" in summary

    def test_handle_agent_death(self, manager):
        # Parent has memories
        for i in range(5):
            manager.record_discovery(1, i, f"resource_{i}", (100, 100))
        manager.flush_pending()

        # Parent dies, child inherits
        inherited = manager.handle_agent_death(
            agent_id=1,
            child_ids=[2],
            current_tick=100,
        )

        assert inherited > 0

        # Child can recall inherited memories
        summaries = manager.recall_for_prompt(2, 100, [])
        assert any("Story from parent" in s for s in summaries)

    def test_stats(self, manager):
        manager.record_interaction(1, 2, 100, "test", "ok", 0.0)
        manager.flush_pending()

        stats = manager.get_stats()

        assert stats['total_stored'] >= 1
        assert 'pending_memories' in stats
