"""Configuration management"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class SimConfig:
    """Simulation configuration with defaults"""

    # World
    WORLD_SIZE: float = 10000.0
    CELL_SIZE: float = 50.0

    # Agents
    INITIAL_AGENT_COUNT: int = 100000
    MAX_AGENTS: int = 1100000

    # Lifecycle
    MAX_AGE: int = 30000
    MATURITY_AGE: int = 5000

    # Energy
    STARTING_ENERGY: float = 100.0
    MAX_ENERGY: float = 100.0
    BASE_ENERGY_DRAIN: float = 0.1
    REPRODUCTION_ENERGY_THRESHOLD: float = 50.0
    REPRODUCTION_ENERGY_COST: float = 30.0
    REPRODUCTION_COOLDOWN: int = 100
    CHILD_STARTING_ENERGY: float = 50.0

    # FSM
    HUNGER_THRESHOLD: float = 30.0
    SATIATED_THRESHOLD: float = 70.0
    REST_THRESHOLD: float = 90.0
    CRITICAL_THRESHOLD: float = 10.0

    # Movement
    BASE_SPEED: float = 1.0
    FLEE_MULTIPLIER: float = 1.5
    VISION_RANGE: float = 50.0
    INTERACTION_RANGE: float = 10.0

    # Simulation
    SEED: Optional[int] = 42
    TICK_RATE_TARGET: int = 100
    CHECKPOINT_INTERVAL: int = 1000
    CHECKPOINT_DIR: str = "checkpoints"
    MAX_CHECKPOINTS: int = 10
    LOG_INTERVAL: int = 100
    EVENT_LOG_DIR: str = "logs/events"
    BATCH_SIZE: int = 10000

    # Ray
    RAY_ENABLED: bool = True
    RAY_NUM_WORKERS: int = 8

    # Resources
    RESOURCES_ENABLED: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> 'SimConfig':
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config = cls()

        # Map YAML structure to flat config
        if 'world' in data:
            config.WORLD_SIZE = data['world'].get('size', config.WORLD_SIZE)
            config.CELL_SIZE = data['world'].get('cell_size', config.CELL_SIZE)

        if 'agents' in data:
            a = data['agents']
            config.INITIAL_AGENT_COUNT = a.get('initial_count', config.INITIAL_AGENT_COUNT)
            config.MAX_AGENTS = a.get('max_count', config.MAX_AGENTS)
            config.MAX_AGE = a.get('max_age', config.MAX_AGE)
            config.MATURITY_AGE = a.get('maturity_age', config.MATURITY_AGE)
            config.STARTING_ENERGY = a.get('starting_energy', config.STARTING_ENERGY)
            config.MAX_ENERGY = a.get('max_energy', config.MAX_ENERGY)
            config.BASE_ENERGY_DRAIN = a.get('base_energy_drain', config.BASE_ENERGY_DRAIN)
            config.REPRODUCTION_ENERGY_THRESHOLD = a.get('reproduction_energy_threshold', config.REPRODUCTION_ENERGY_THRESHOLD)
            config.REPRODUCTION_ENERGY_COST = a.get('reproduction_energy_cost', config.REPRODUCTION_ENERGY_COST)
            config.REPRODUCTION_COOLDOWN = a.get('reproduction_cooldown', config.REPRODUCTION_COOLDOWN)
            config.CHILD_STARTING_ENERGY = a.get('child_starting_energy', config.CHILD_STARTING_ENERGY)

        if 'fsm' in data:
            f = data['fsm']
            config.HUNGER_THRESHOLD = f.get('hunger_threshold', config.HUNGER_THRESHOLD)
            config.SATIATED_THRESHOLD = f.get('satiated_threshold', config.SATIATED_THRESHOLD)
            config.REST_THRESHOLD = f.get('rest_threshold', config.REST_THRESHOLD)
            config.CRITICAL_THRESHOLD = f.get('critical_threshold', config.CRITICAL_THRESHOLD)

        if 'movement' in data:
            m = data['movement']
            config.BASE_SPEED = m.get('base_speed', config.BASE_SPEED)
            config.FLEE_MULTIPLIER = m.get('flee_multiplier', config.FLEE_MULTIPLIER)
            config.VISION_RANGE = m.get('vision_range', config.VISION_RANGE)
            config.INTERACTION_RANGE = m.get('interaction_range', config.INTERACTION_RANGE)

        if 'simulation' in data:
            s = data['simulation']
            config.SEED = s.get('seed', config.SEED)
            config.TICK_RATE_TARGET = s.get('tick_rate_target', config.TICK_RATE_TARGET)
            config.CHECKPOINT_INTERVAL = s.get('checkpoint_interval', config.CHECKPOINT_INTERVAL)
            config.CHECKPOINT_DIR = s.get('checkpoint_dir', config.CHECKPOINT_DIR)
            config.MAX_CHECKPOINTS = s.get('max_checkpoints', config.MAX_CHECKPOINTS)
            config.LOG_INTERVAL = s.get('log_interval', config.LOG_INTERVAL)
            config.EVENT_LOG_DIR = s.get('event_log_dir', config.EVENT_LOG_DIR)
            config.BATCH_SIZE = s.get('batch_size', config.BATCH_SIZE)

        if 'ray' in data:
            r = data['ray']
            config.RAY_ENABLED = r.get('enabled', config.RAY_ENABLED)
            config.RAY_NUM_WORKERS = r.get('num_workers', config.RAY_NUM_WORKERS)

        if 'resources' in data:
            config.RESOURCES_ENABLED = data['resources'].get('enabled', config.RESOURCES_ENABLED)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# Global config instance
_config: Optional[SimConfig] = None


def get_config() -> SimConfig:
    """Get global configuration"""
    global _config
    if _config is None:
        _config = SimConfig()
    return _config


def load_config(path: str) -> SimConfig:
    """Load and set global configuration"""
    global _config
    _config = SimConfig.from_yaml(path)
    return _config


def set_config(config: SimConfig) -> None:
    """Set global configuration"""
    global _config
    _config = config
