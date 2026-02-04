"""Memory Manager - High-level interface for agent memories"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from .schema import Memory, MemoryType, Relationship, create_memory_id
from .store import MemoryStore
from .embeddings import create_embedding_model


class MemoryManager:
    """
    High-level interface for managing agent memories.

    Handles:
    - Memory creation from events
    - Recall for LLM prompts
    - Inheritance on agent death
    - Relationship tracking
    """

    def __init__(
        self,
        db_path: str = "./memory_db",
        use_mock: bool = False,
    ):
        self.store = MemoryStore(db_path=db_path, use_mock=use_mock)
        self.use_mock = use_mock

        # Pending memories to batch process
        self.pending_memories: List[Memory] = []

        # Cache for recent queries
        self.recall_cache: Dict[int, List[str]] = {}
        self.cache_tick: int = -1

    def record_interaction(
        self,
        agent_id: int,
        other_agent_id: int,
        tick: int,
        action: str,
        outcome: str,
        valence: float,
        location: Optional[Tuple[float, float]] = None,
    ) -> Memory:
        """Record an interaction between two agents"""
        summary = f"Met Human-{other_agent_id} and {action}. {outcome}"

        memory = Memory(
            id=create_memory_id(agent_id, tick, f"int_{other_agent_id}"),
            agent_id=agent_id,
            tick=tick,
            memory_type=MemoryType.INTERACTION,
            summary=summary,
            emotional_valence=valence,
            importance=0.5 + abs(valence) * 0.3,  # More emotional = more important
            other_agent_id=other_agent_id,
            location_x=location[0] if location else None,
            location_y=location[1] if location else None,
        )

        self.pending_memories.append(memory)
        return memory

    def record_trade(
        self,
        agent_id: int,
        other_agent_id: int,
        tick: int,
        gave: Dict[str, int],
        received: Dict[str, int],
        location: Optional[Tuple[float, float]] = None,
    ) -> Memory:
        """Record a trade event"""
        gave_str = ", ".join(f"{v} {k}" for k, v in gave.items()) or "nothing"
        recv_str = ", ".join(f"{v} {k}" for k, v in received.items()) or "nothing"
        summary = f"Traded with Human-{other_agent_id}: gave {gave_str}, received {recv_str}"

        # Positive if received more value
        gave_value = sum(gave.values())
        recv_value = sum(received.values())
        valence = 0.3 if recv_value >= gave_value else -0.1

        memory = Memory(
            id=create_memory_id(agent_id, tick, f"trade_{other_agent_id}"),
            agent_id=agent_id,
            tick=tick,
            memory_type=MemoryType.TRADE,
            summary=summary,
            emotional_valence=valence,
            importance=0.6,
            other_agent_id=other_agent_id,
            location_x=location[0] if location else None,
            location_y=location[1] if location else None,
        )

        self.pending_memories.append(memory)
        return memory

    def record_conflict(
        self,
        agent_id: int,
        other_agent_id: int,
        tick: int,
        won: bool,
        location: Optional[Tuple[float, float]] = None,
    ) -> Memory:
        """Record a conflict event"""
        outcome = "won" if won else "lost"
        summary = f"Fought with Human-{other_agent_id} and {outcome}"
        valence = 0.2 if won else -0.5

        memory = Memory(
            id=create_memory_id(agent_id, tick, f"fight_{other_agent_id}"),
            agent_id=agent_id,
            tick=tick,
            memory_type=MemoryType.CONFLICT,
            summary=summary,
            emotional_valence=valence,
            importance=0.8,  # Conflicts are very memorable
            other_agent_id=other_agent_id,
            location_x=location[0] if location else None,
            location_y=location[1] if location else None,
        )

        self.pending_memories.append(memory)
        return memory

    def record_discovery(
        self,
        agent_id: int,
        tick: int,
        what: str,
        location: Tuple[float, float],
    ) -> Memory:
        """Record a discovery"""
        summary = f"Discovered {what} at this location"

        memory = Memory(
            id=create_memory_id(agent_id, tick, f"disc_{hash(what) % 10000}"),
            agent_id=agent_id,
            tick=tick,
            memory_type=MemoryType.DISCOVERY,
            summary=summary,
            emotional_valence=0.4,
            importance=0.7,
            location_x=location[0],
            location_y=location[1],
        )

        self.pending_memories.append(memory)
        return memory

    def record_birth(
        self,
        agent_id: int,
        tick: int,
        child_id: int,
        partner_id: int,
        location: Optional[Tuple[float, float]] = None,
    ) -> Memory:
        """Record witnessing/participating in birth"""
        summary = f"Had a child (Human-{child_id}) with Human-{partner_id}"

        memory = Memory(
            id=create_memory_id(agent_id, tick, f"birth_{child_id}"),
            agent_id=agent_id,
            tick=tick,
            memory_type=MemoryType.BIRTH,
            summary=summary,
            emotional_valence=0.8,
            importance=0.9,  # Very important
            other_agent_id=partner_id,
            location_x=location[0] if location else None,
            location_y=location[1] if location else None,
        )

        self.pending_memories.append(memory)
        return memory

    def record_death(
        self,
        agent_id: int,
        tick: int,
        deceased_id: int,
        cause: str,
        location: Optional[Tuple[float, float]] = None,
    ) -> Memory:
        """Record witnessing a death"""
        summary = f"Witnessed Human-{deceased_id} die from {cause}"

        memory = Memory(
            id=create_memory_id(agent_id, tick, f"death_{deceased_id}"),
            agent_id=agent_id,
            tick=tick,
            memory_type=MemoryType.DEATH,
            summary=summary,
            emotional_valence=-0.6,
            importance=0.8,
            other_agent_id=deceased_id,
            location_x=location[0] if location else None,
            location_y=location[1] if location else None,
        )

        self.pending_memories.append(memory)
        return memory

    def flush_pending(self) -> int:
        """Store all pending memories"""
        count = len(self.pending_memories)
        for memory in self.pending_memories:
            self.store.store(memory)
        self.pending_memories.clear()
        return count

    def recall_for_prompt(
        self,
        agent_id: int,
        current_tick: int,
        nearby_agent_ids: List[int],
        limit: int = 5,
    ) -> List[str]:
        """
        Get memories relevant for LLM prompt.

        Returns list of memory summaries.
        """
        # Check cache
        if current_tick == self.cache_tick and agent_id in self.recall_cache:
            return self.recall_cache[agent_id]

        memories = []

        # Get recent memories
        recent = self.store.query_by_agent(agent_id, limit=3)
        memories.extend(recent)

        # Get memories about nearby agents
        for other_id in nearby_agent_ids[:3]:
            about = self.store.query_about_agent(agent_id, other_id, limit=1)
            memories.extend(about)

        # Deduplicate and sort by importance
        seen_ids = set()
        unique = []
        for m in memories:
            if m.id not in seen_ids:
                seen_ids.add(m.id)
                unique.append(m)

        unique.sort(key=lambda m: m.importance, reverse=True)

        # Format as strings
        summaries = [m.summary for m in unique[:limit]]

        # Update cache
        if current_tick != self.cache_tick:
            self.recall_cache.clear()
            self.cache_tick = current_tick
        self.recall_cache[agent_id] = summaries

        return summaries

    def get_relationship_summary(
        self,
        agent_id: int,
        other_agent_id: int,
    ) -> Optional[str]:
        """Get a summary of relationship with another agent"""
        rel = self.store.get_relationship(agent_id, other_agent_id)
        if not rel:
            return None

        if rel.trust_score > 0.3:
            feeling = "friendly"
        elif rel.trust_score < -0.3:
            feeling = "hostile"
        else:
            feeling = "neutral"

        if rel.familiarity > 0.5:
            familiarity = "well-known"
        elif rel.familiarity > 0.2:
            familiarity = "acquaintance"
        else:
            familiarity = "barely known"

        return f"{familiarity}, {feeling} ({rel.interaction_count} interactions)"

    def handle_agent_death(
        self,
        agent_id: int,
        child_ids: List[int],
        current_tick: int,
    ) -> int:
        """
        Handle memory inheritance when agent dies.

        Returns total memories inherited by children.
        """
        total_inherited = 0

        for child_id in child_ids:
            inherited = self.store.inherit_memories(
                parent_id=agent_id,
                child_id=child_id,
                current_tick=current_tick,
                inherit_fraction=0.3,
            )
            total_inherited += inherited

        # Optionally delete parent memories to save space
        # self.store.delete_agent_memories(agent_id)

        return total_inherited

    def get_stats(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        store_stats = self.store.get_stats()
        store_stats['pending_memories'] = len(self.pending_memories)
        return store_stats
