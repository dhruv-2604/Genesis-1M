"""Agent and World State Management"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Tuple, Optional, List
import numpy as np
import json
import struct


class AgentTier(IntEnum):
    """Agent complexity tiers"""
    TIER3 = 3  # FSM-only (99.9% of agents)
    TIER2 = 2  # Weighted heuristics (personality-biased FSM)
    TIER1 = 1  # LLM-driven (rare, interesting situations)


class FSMState(IntEnum):
    """Finite State Machine states for Tier 3 agents"""
    WANDER = 0
    SEEK_FOOD = 1
    FLEE = 2
    SEEK_MATE = 3
    REST = 4
    SEEK_WATER = 5
    RETURN_HOME = 6


# Gene indices for trait lookup
class GeneIndex(IntEnum):
    # Physical traits (affect FSM mechanics)
    SPEED = 0           # Movement speed multiplier (0.5 - 1.5)
    METABOLISM = 1      # Energy consumption rate (0.5 - 1.5)
    FERTILITY = 2       # Reproduction success rate (0 - 1)
    VISION_RANGE = 3    # Detection radius multiplier (0.5 - 1.5)
    STRENGTH = 4        # Combat/hunting effectiveness (0 - 1)

    # Personality traits (injected into LLM prompts)
    AGGRESSION = 5      # 0 = pacifist, 1 = violent/territorial
    SOCIABILITY = 6     # 0 = loner, 1 = seeks groups
    ALTRUISM = 7        # 0 = selfish, 1 = self-sacrificing
    GREED = 8           # 0 = generous/shares, 1 = hoards everything
    CURIOSITY = 9       # 0 = routine/conservative, 1 = explores/experiments
    TRUST = 10          # 0 = paranoid/suspicious, 1 = naive/trusting
    INTELLIGENCE = 11   # Promotion priority boost (0 - 1)

    NUM_GENES = 12


@dataclass
class AgentState:
    """Core agent state - optimized for vectorized operations"""
    id: int
    x: float
    y: float
    energy: float = 100.0
    age: int = 0

    # Genes - heritable traits that drive selection pressure
    genes: np.ndarray = field(default_factory=lambda: np.random.uniform(0.5, 1.0, GeneIndex.NUM_GENES))

    # Inventory - resources carried
    inventory: Dict[str, int] = field(default_factory=dict)

    # Memory pointer (None for Tier 3, populated for Tier 1/2)
    memory_id: Optional[str] = None

    # Current state
    tier: AgentTier = AgentTier.TIER3
    fsm_state: FSMState = FSMState.WANDER

    # Lifecycle
    reproductive_cooldown: int = 0
    parent_ids: Tuple[int, int] = (-1, -1)

    # Social tracking (lightweight, for promotion scoring)
    known_agents: set = field(default_factory=set)
    home_x: Optional[float] = None
    home_y: Optional[float] = None

    # Target tracking for current action
    target_id: Optional[int] = None
    target_x: Optional[float] = None
    target_y: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.genes, list):
            self.genes = np.array(self.genes, dtype=np.float32)
        if not isinstance(self.known_agents, set):
            self.known_agents = set(self.known_agents) if self.known_agents else set()

    @property
    def speed(self) -> float:
        return self.genes[GeneIndex.SPEED]

    @property
    def metabolism(self) -> float:
        return self.genes[GeneIndex.METABOLISM]

    @property
    def sociability(self) -> float:
        return self.genes[GeneIndex.SOCIABILITY]

    @property
    def aggression(self) -> float:
        return self.genes[GeneIndex.AGGRESSION]

    @property
    def altruism(self) -> float:
        return self.genes[GeneIndex.ALTRUISM]

    @property
    def greed(self) -> float:
        return self.genes[GeneIndex.GREED]

    @property
    def curiosity(self) -> float:
        return self.genes[GeneIndex.CURIOSITY]

    @property
    def trust(self) -> float:
        return self.genes[GeneIndex.TRUST]

    @property
    def is_alive(self) -> bool:
        return self.energy > 0

    @property
    def can_reproduce(self) -> bool:
        from ..config import SimConfig
        return (
            self.age >= SimConfig.MATURITY_AGE and
            self.reproductive_cooldown == 0 and
            self.energy >= SimConfig.REPRODUCTION_ENERGY_THRESHOLD
        )

    def to_dict(self) -> dict:
        """Serialize for checkpointing"""
        return {
            'id': int(self.id),
            'x': float(self.x),
            'y': float(self.y),
            'energy': float(self.energy),
            'age': int(self.age),
            'genes': self.genes.tolist(),
            'inventory': self.inventory,
            'memory_id': self.memory_id,
            'tier': int(self.tier),
            'fsm_state': int(self.fsm_state),
            'reproductive_cooldown': int(self.reproductive_cooldown),
            'parent_ids': [int(p) for p in self.parent_ids],
            'known_agents': [int(a) for a in self.known_agents],
            'home_x': float(self.home_x) if self.home_x is not None else None,
            'home_y': float(self.home_y) if self.home_y is not None else None,
            'target_id': int(self.target_id) if self.target_id is not None else None,
            'target_x': float(self.target_x) if self.target_x is not None else None,
            'target_y': float(self.target_y) if self.target_y is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AgentState':
        """Deserialize from checkpoint"""
        return cls(
            id=data['id'],
            x=data['x'],
            y=data['y'],
            energy=data['energy'],
            age=data['age'],
            genes=np.array(data['genes'], dtype=np.float32),
            inventory=data['inventory'],
            memory_id=data.get('memory_id'),
            tier=AgentTier(data['tier']),
            fsm_state=FSMState(data['fsm_state']),
            reproductive_cooldown=data['reproductive_cooldown'],
            parent_ids=tuple(data['parent_ids']),
            known_agents=set(data.get('known_agents', [])),
            home_x=data.get('home_x'),
            home_y=data.get('home_y'),
            target_id=data.get('target_id'),
            target_x=data.get('target_x'),
            target_y=data.get('target_y'),
        )


@dataclass
class WorldState:
    """Global world state"""
    tick: int = 0
    world_size: float = 10000.0

    # Agent tracking
    agents: Dict[int, AgentState] = field(default_factory=dict)
    next_agent_id: int = 0

    # Statistics
    total_births: int = 0
    total_deaths: int = 0

    # Resource state is managed by ResourceManager

    def add_agent(self, agent: AgentState) -> None:
        self.agents[agent.id] = agent

    def remove_agent(self, agent_id: int) -> Optional[AgentState]:
        return self.agents.pop(agent_id, None)

    def get_agent(self, agent_id: int) -> Optional[AgentState]:
        return self.agents.get(agent_id)

    def get_new_id(self) -> int:
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        return agent_id

    @property
    def population(self) -> int:
        return len(self.agents)

    def to_dict(self) -> dict:
        """Serialize for checkpointing"""
        return {
            'tick': self.tick,
            'world_size': self.world_size,
            'agents': {str(k): v.to_dict() for k, v in self.agents.items()},
            'next_agent_id': self.next_agent_id,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WorldState':
        """Deserialize from checkpoint"""
        state = cls(
            tick=data['tick'],
            world_size=data['world_size'],
            next_agent_id=data['next_agent_id'],
            total_births=data['total_births'],
            total_deaths=data['total_deaths'],
        )
        for agent_id, agent_data in data['agents'].items():
            state.agents[int(agent_id)] = AgentState.from_dict(agent_data)
        return state


class AgentArrays:
    """
    Vectorized agent data for fast NumPy operations.
    Maintains parallel arrays that can be updated in batch.
    """

    def __init__(self, max_agents: int = 1_100_000):
        self.max_agents = max_agents
        self.count = 0

        # Core arrays
        self.ids = np.zeros(max_agents, dtype=np.int32)
        self.x = np.zeros(max_agents, dtype=np.float32)
        self.y = np.zeros(max_agents, dtype=np.float32)
        self.energy = np.zeros(max_agents, dtype=np.float32)
        self.age = np.zeros(max_agents, dtype=np.int32)
        self.fsm_state = np.zeros(max_agents, dtype=np.int8)
        self.tier = np.zeros(max_agents, dtype=np.int8)

        # Genes matrix: (max_agents, NUM_GENES)
        self.genes = np.zeros((max_agents, GeneIndex.NUM_GENES), dtype=np.float32)

        # Lifecycle
        self.reproductive_cooldown = np.zeros(max_agents, dtype=np.int32)
        self.alive = np.zeros(max_agents, dtype=np.bool_)

        # Index mapping: agent_id -> array index
        self.id_to_index: Dict[int, int] = {}

        # Free slots for reuse
        self.free_slots: List[int] = []

    def add_agent(self, agent: AgentState) -> int:
        """Add agent to arrays, return array index"""
        if self.free_slots:
            idx = self.free_slots.pop()
        else:
            idx = self.count
            self.count += 1

        if idx >= self.max_agents:
            raise RuntimeError(f"Exceeded max agents: {self.max_agents}")

        self.ids[idx] = agent.id
        self.x[idx] = agent.x
        self.y[idx] = agent.y
        self.energy[idx] = agent.energy
        self.age[idx] = agent.age
        self.fsm_state[idx] = int(agent.fsm_state)
        self.tier[idx] = int(agent.tier)
        self.genes[idx] = agent.genes
        self.reproductive_cooldown[idx] = agent.reproductive_cooldown
        self.alive[idx] = True

        self.id_to_index[agent.id] = idx
        return idx

    def remove_agent(self, agent_id: int) -> None:
        """Mark agent as dead, slot available for reuse"""
        if agent_id not in self.id_to_index:
            return
        idx = self.id_to_index.pop(agent_id)
        self.alive[idx] = False
        self.free_slots.append(idx)

    def sync_to_state(self, world_state: WorldState) -> None:
        """Sync array data back to AgentState objects"""
        for agent_id, idx in self.id_to_index.items():
            if not self.alive[idx]:
                continue
            agent = world_state.agents.get(agent_id)
            if agent:
                agent.x = float(self.x[idx])
                agent.y = float(self.y[idx])
                agent.energy = float(self.energy[idx])
                agent.age = int(self.age[idx])
                agent.fsm_state = FSMState(self.fsm_state[idx])
                agent.reproductive_cooldown = int(self.reproductive_cooldown[idx])

    def sync_from_state(self, world_state: WorldState) -> None:
        """Sync from AgentState objects to arrays"""
        for agent_id, agent in world_state.agents.items():
            if agent_id not in self.id_to_index:
                self.add_agent(agent)
            else:
                idx = self.id_to_index[agent_id]
                self.x[idx] = agent.x
                self.y[idx] = agent.y
                self.energy[idx] = agent.energy
                self.age[idx] = agent.age
                self.fsm_state[idx] = int(agent.fsm_state)
                self.reproductive_cooldown[idx] = agent.reproductive_cooldown

    def get_alive_mask(self) -> np.ndarray:
        """Return boolean mask of alive agents"""
        return self.alive[:self.count]

    def get_alive_indices(self) -> np.ndarray:
        """Return indices of alive agents"""
        return np.where(self.alive[:self.count])[0]
