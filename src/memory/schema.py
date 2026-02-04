"""Memory Schema and Data Models"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum, auto
import numpy as np


class MemoryType(Enum):
    """Types of memories agents can form"""
    INTERACTION = auto()      # Met/talked with another agent
    TRADE = auto()            # Exchanged resources
    CONFLICT = auto()         # Fight or flight
    DISCOVERY = auto()        # Found something new
    BIRTH = auto()            # Witnessed birth
    DEATH = auto()            # Witnessed death
    INHERITED = auto()        # Story from parent
    OBSERVATION = auto()      # General observation


@dataclass
class Memory:
    """
    Single memory record for an agent.

    Memories are stored in LanceDB with embeddings for semantic search.
    """
    id: str                              # Unique ID (agent_id + timestamp)
    agent_id: int                        # Owner of this memory
    tick: int                            # When this happened
    memory_type: MemoryType              # Category
    summary: str                         # Human-readable description
    embedding: Optional[np.ndarray] = None  # 384-dim vector (all-MiniLM-L6-v2)

    # Emotional and relationship tracking
    emotional_valence: float = 0.0       # -1 (negative) to 1 (positive)
    importance: float = 0.5              # 0-1, affects retention

    # Related entities
    other_agent_id: Optional[int] = None # Other agent involved
    location_x: Optional[float] = None   # Where it happened
    location_y: Optional[float] = None

    # Inheritance tracking
    inherited_from: Optional[int] = None # Parent who passed this down
    generation: int = 0                  # How many generations old

    # Access tracking
    access_count: int = 0                # Times recalled
    last_accessed: int = 0               # Last tick accessed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for LanceDB storage"""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'tick': self.tick,
            'memory_type': self.memory_type.name,
            'summary': self.summary,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'emotional_valence': self.emotional_valence,
            'importance': self.importance,
            'other_agent_id': self.other_agent_id,
            'location_x': self.location_x,
            'location_y': self.location_y,
            'inherited_from': self.inherited_from,
            'generation': self.generation,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dict"""
        embedding = None
        if data.get('embedding'):
            embedding = np.array(data['embedding'], dtype=np.float32)

        return cls(
            id=data['id'],
            agent_id=data['agent_id'],
            tick=data['tick'],
            memory_type=MemoryType[data['memory_type']],
            summary=data['summary'],
            embedding=embedding,
            emotional_valence=data.get('emotional_valence', 0.0),
            importance=data.get('importance', 0.5),
            other_agent_id=data.get('other_agent_id'),
            location_x=data.get('location_x'),
            location_y=data.get('location_y'),
            inherited_from=data.get('inherited_from'),
            generation=data.get('generation', 0),
            access_count=data.get('access_count', 0),
            last_accessed=data.get('last_accessed', 0),
        )


@dataclass
class Relationship:
    """
    Tracks relationship between two agents based on memories.
    """
    agent_id: int
    other_agent_id: int

    # Aggregate stats from memories
    interaction_count: int = 0
    total_valence: float = 0.0           # Sum of emotional valences
    last_interaction_tick: int = 0

    # Computed relationship score
    @property
    def trust_score(self) -> float:
        """Average emotional valence of interactions"""
        if self.interaction_count == 0:
            return 0.0
        return self.total_valence / self.interaction_count

    @property
    def familiarity(self) -> float:
        """How well they know each other (0-1)"""
        # Asymptotic approach to 1.0
        return 1.0 - (1.0 / (1.0 + self.interaction_count * 0.1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'agent_id': self.agent_id,
            'other_agent_id': self.other_agent_id,
            'interaction_count': self.interaction_count,
            'total_valence': self.total_valence,
            'last_interaction_tick': self.last_interaction_tick,
        }


def create_memory_id(agent_id: int, tick: int, suffix: str = "") -> str:
    """Generate unique memory ID"""
    return f"{agent_id}_{tick}_{suffix}"
