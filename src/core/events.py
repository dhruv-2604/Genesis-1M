"""Event logging system for post-hoc analysis and content creation"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import gzip
import time


class EventType(Enum):
    """Types of events to log"""
    BIRTH = auto()
    DEATH = auto()
    TRADE = auto()
    CONFLICT = auto()
    DISCOVERY = auto()
    MILESTONE = auto()
    EAT = auto()
    REPRODUCE = auto()
    STATE_CHANGE = auto()
    PROMOTION = auto()
    DEMOTION = auto()


@dataclass
class Event:
    """Single event record"""
    type: EventType
    tick: int
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        # Convert numpy types to Python native for JSON serialization
        import numpy as np

        def convert(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        return {
            'type': self.type.name,
            'tick': int(self.tick),
            'data': convert(self.data),
            'timestamp': self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Event':
        return cls(
            type=EventType[data['type']],
            tick=data['tick'],
            data=data['data'],
            timestamp=data.get('timestamp', 0),
        )


class EventLogger:
    """
    Batched event logger for efficient disk writes.

    Events are buffered in memory and flushed to disk periodically
    or when buffer is full.
    """

    def __init__(
        self,
        log_dir: str,
        buffer_size: int = 10000,
        compress: bool = True
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_size = buffer_size
        self.compress = compress

        self.buffer: List[Event] = []
        self.file_counter = 0

        # Statistics
        self.total_events = 0
        self.events_by_type: Dict[EventType, int] = {t: 0 for t in EventType}

    def log(self, event_type: EventType, tick: int, **data) -> None:
        """Log an event"""
        event = Event(type=event_type, tick=tick, data=data)
        self.buffer.append(event)
        self.total_events += 1
        self.events_by_type[event_type] += 1

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def log_birth(
        self,
        tick: int,
        child_id: int,
        parent_ids: Tuple[int, int],
        x: float,
        y: float,
        genes: List[float]
    ) -> None:
        """Log a birth event"""
        self.log(
            EventType.BIRTH,
            tick,
            child_id=child_id,
            parent1_id=parent_ids[0],
            parent2_id=parent_ids[1],
            x=x,
            y=y,
            genes=genes,
        )

    def log_death(
        self,
        tick: int,
        agent_id: int,
        cause: str,
        age: int,
        x: float,
        y: float
    ) -> None:
        """Log a death event"""
        self.log(
            EventType.DEATH,
            tick,
            agent_id=agent_id,
            cause=cause,
            age=age,
            x=x,
            y=y,
        )

    def log_trade(
        self,
        tick: int,
        agent1_id: int,
        agent2_id: int,
        items1: Dict[str, int],
        items2: Dict[str, int]
    ) -> None:
        """Log a trade event"""
        self.log(
            EventType.TRADE,
            tick,
            agent1_id=agent1_id,
            agent2_id=agent2_id,
            agent1_gives=items1,
            agent2_gives=items2,
        )

    def log_conflict(
        self,
        tick: int,
        aggressor_id: int,
        defender_id: int,
        outcome: str,
        x: float,
        y: float
    ) -> None:
        """Log a conflict event"""
        self.log(
            EventType.CONFLICT,
            tick,
            aggressor_id=aggressor_id,
            defender_id=defender_id,
            outcome=outcome,
            x=x,
            y=y,
        )

    def log_discovery(
        self,
        tick: int,
        agent_id: int,
        discovery_type: str,
        x: float,
        y: float
    ) -> None:
        """Log a discovery event (first to find something)"""
        self.log(
            EventType.DISCOVERY,
            tick,
            agent_id=agent_id,
            discovery_type=discovery_type,
            x=x,
            y=y,
        )

    def log_milestone(
        self,
        tick: int,
        milestone_type: str,
        details: Dict[str, Any]
    ) -> None:
        """Log a civilization milestone"""
        self.log(
            EventType.MILESTONE,
            tick,
            milestone_type=milestone_type,
            **details,
        )

    def log_eat(
        self,
        tick: int,
        agent_id: int,
        resource_type: str,
        amount: float,
        x: float,
        y: float
    ) -> None:
        """Log eating/foraging event"""
        self.log(
            EventType.EAT,
            tick,
            agent_id=agent_id,
            resource_type=resource_type,
            amount=amount,
            x=x,
            y=y,
        )

    def log_promotion(
        self,
        tick: int,
        agent_id: int,
        from_tier: int,
        to_tier: int,
        reason: str
    ) -> None:
        """Log tier promotion"""
        self.log(
            EventType.PROMOTION,
            tick,
            agent_id=agent_id,
            from_tier=from_tier,
            to_tier=to_tier,
            reason=reason,
        )

    def flush(self) -> None:
        """Write buffer to disk"""
        if not self.buffer:
            return

        filename = f"events_{self.file_counter:08d}"
        if self.compress:
            filepath = self.log_dir / f"{filename}.jsonl.gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                for event in self.buffer:
                    f.write(json.dumps(event.to_dict()) + '\n')
        else:
            filepath = self.log_dir / f"{filename}.jsonl"
            with open(filepath, 'w', encoding='utf-8') as f:
                for event in self.buffer:
                    f.write(json.dumps(event.to_dict()) + '\n')

        self.buffer.clear()
        self.file_counter += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'total_events': self.total_events,
            'events_by_type': {t.name: c for t, c in self.events_by_type.items()},
            'files_written': self.file_counter,
            'buffer_size': len(self.buffer),
        }

    def close(self) -> None:
        """Flush remaining events and close"""
        self.flush()


class EventReader:
    """Read events from log files"""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)

    def read_all(self) -> List[Event]:
        """Read all events from all files"""
        events = []

        for filepath in sorted(self.log_dir.glob("events_*.jsonl*")):
            if filepath.suffix == '.gz':
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        events.append(Event.from_dict(json.loads(line)))
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        events.append(Event.from_dict(json.loads(line)))

        return events

    def read_by_type(self, event_type: EventType) -> List[Event]:
        """Read only events of a specific type"""
        return [e for e in self.read_all() if e.type == event_type]

    def read_tick_range(self, start_tick: int, end_tick: int) -> List[Event]:
        """Read events within a tick range"""
        return [e for e in self.read_all() if start_tick <= e.tick <= end_tick]

    def get_lineage(self, agent_id: int) -> Dict[str, Any]:
        """Reconstruct an agent's lineage from birth/death events"""
        births = {e.data['child_id']: e for e in self.read_by_type(EventType.BIRTH)}
        deaths = {e.data['agent_id']: e for e in self.read_by_type(EventType.DEATH)}

        lineage = {'agent_id': agent_id, 'ancestors': [], 'descendants': []}

        # Find ancestors
        current_id = agent_id
        while current_id in births:
            birth = births[current_id]
            lineage['ancestors'].append({
                'parent1': birth.data['parent1_id'],
                'parent2': birth.data['parent2_id'],
                'tick': birth.tick,
            })
            current_id = birth.data['parent1_id']  # Follow one parent line

        # Find descendants
        for child_id, birth in births.items():
            if birth.data['parent1_id'] == agent_id or birth.data['parent2_id'] == agent_id:
                lineage['descendants'].append({
                    'child_id': child_id,
                    'tick': birth.tick,
                })

        return lineage
