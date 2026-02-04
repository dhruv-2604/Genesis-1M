"""Resource management system"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import numpy as np


class ResourceType(Enum):
    """Types of resources in the world"""
    BERRIES = auto()
    MEAT = auto()
    FISH = auto()
    WATER = auto()
    FLINT = auto()
    WOOD = auto()


@dataclass
class ResourceConfig:
    """Configuration for a resource type"""
    spawn_rate: float = 0.01        # Per cell per tick
    max_per_cell: int = 10          # Carrying capacity
    energy_value: float = 20.0      # Energy gained from consuming
    requires_tool: bool = False      # Needs tool to harvest
    tool_type: Optional[str] = None  # Which tool is needed
    respawn_time: int = 100         # Ticks to respawn after depleted


DEFAULT_RESOURCE_CONFIGS = {
    ResourceType.BERRIES: ResourceConfig(
        spawn_rate=0.01,
        max_per_cell=10,
        energy_value=20.0,
        requires_tool=False,
    ),
    ResourceType.MEAT: ResourceConfig(
        spawn_rate=0.001,
        max_per_cell=3,
        energy_value=50.0,
        requires_tool=True,
        tool_type='flint',
    ),
    ResourceType.FISH: ResourceConfig(
        spawn_rate=0.005,
        max_per_cell=5,
        energy_value=30.0,
        requires_tool=False,
    ),
    ResourceType.WATER: ResourceConfig(
        spawn_rate=0.0,  # Fixed locations
        max_per_cell=100,
        energy_value=10.0,
        requires_tool=False,
    ),
    ResourceType.FLINT: ResourceConfig(
        spawn_rate=0.0001,
        max_per_cell=2,
        energy_value=0.0,  # Tool, not food
        requires_tool=False,
    ),
    ResourceType.WOOD: ResourceConfig(
        spawn_rate=0.002,
        max_per_cell=20,
        energy_value=0.0,  # Building material
        requires_tool=False,
    ),
}


@dataclass
class ResourceCell:
    """Resource state for a single cell"""
    amounts: Dict[ResourceType, float] = field(default_factory=dict)
    respawn_timers: Dict[ResourceType, int] = field(default_factory=dict)

    def get(self, resource_type: ResourceType) -> float:
        return self.amounts.get(resource_type, 0.0)

    def add(self, resource_type: ResourceType, amount: float, max_amount: float) -> None:
        current = self.amounts.get(resource_type, 0.0)
        self.amounts[resource_type] = min(current + amount, max_amount)

    def take(self, resource_type: ResourceType, amount: float) -> float:
        """Take up to amount, return actual amount taken"""
        current = self.amounts.get(resource_type, 0.0)
        taken = min(current, amount)
        self.amounts[resource_type] = current - taken
        return taken


class ResourceManager:
    """
    Manages all resources in the world.

    Resources are stored per-cell and regenerate over time.
    Integrates with terrain for spawn rate modifiers.
    """

    def __init__(
        self,
        world_size: float,
        cell_size: float,
        configs: Optional[Dict[ResourceType, ResourceConfig]] = None,
        seed: int = 42
    ):
        self.world_size = world_size
        self.cell_size = cell_size
        self.num_cells = int(np.ceil(world_size / cell_size))

        self.configs = configs or DEFAULT_RESOURCE_CONFIGS
        self.rng = np.random.default_rng(seed)

        # Resource storage: (cell_x, cell_y) -> ResourceCell
        self.cells: Dict[Tuple[int, int], ResourceCell] = defaultdict(ResourceCell)

        # Track cells with resources for efficient iteration
        self.active_cells: Set[Tuple[int, int]] = set()

        # Statistics
        self.total_harvested: Dict[ResourceType, float] = {t: 0.0 for t in ResourceType}
        self.total_spawned: Dict[ResourceType, float] = {t: 0.0 for t in ResourceType}

    def _get_cell_coords(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to cell coordinates"""
        cx = int(x / self.cell_size) % self.num_cells
        cy = int(y / self.cell_size) % self.num_cells
        return (cx, cy)

    def initialize_resources(
        self,
        terrain_generator=None,
        initial_density: float = 0.3
    ) -> None:
        """Spawn initial resources across the world"""
        for cx in range(self.num_cells):
            for cy in range(self.num_cells):
                # World coordinates for this cell (center)
                wx = (cx + 0.5) * self.cell_size
                wy = (cy + 0.5) * self.cell_size

                for resource_type, config in self.configs.items():
                    if config.spawn_rate == 0 and resource_type != ResourceType.WATER:
                        continue

                    # Get terrain multiplier if terrain generator provided
                    multiplier = 1.0
                    if terrain_generator:
                        multiplier = terrain_generator.get_resource_multiplier(
                            wx, wy, resource_type.name.lower()
                        )

                    if multiplier <= 0:
                        continue

                    # Probabilistic initial spawn
                    if self.rng.random() < initial_density * multiplier:
                        max_amount = max(1.0, config.max_per_cell * multiplier)
                        initial_amount = self.rng.uniform(1, max_amount)
                        cell = self.cells[(cx, cy)]
                        cell.add(resource_type, initial_amount, config.max_per_cell)
                        self.active_cells.add((cx, cy))
                        self.total_spawned[resource_type] += initial_amount

    def regenerate(self, terrain_generator=None) -> None:
        """Regenerate resources across all cells (call once per tick)"""
        cells_to_check = list(self.active_cells) + [
            (cx, cy)
            for cx in range(0, self.num_cells, 10)
            for cy in range(0, self.num_cells, 10)
        ]

        for cx, cy in cells_to_check:
            wx = (cx + 0.5) * self.cell_size
            wy = (cy + 0.5) * self.cell_size

            cell = self.cells[(cx, cy)]

            for resource_type, config in self.configs.items():
                if config.spawn_rate == 0:
                    continue

                # Check respawn timer
                timer = cell.respawn_timers.get(resource_type, 0)
                if timer > 0:
                    cell.respawn_timers[resource_type] = timer - 1
                    continue

                current = cell.get(resource_type)
                if current >= config.max_per_cell:
                    continue

                # Get terrain multiplier
                multiplier = 1.0
                if terrain_generator:
                    multiplier = terrain_generator.get_resource_multiplier(
                        wx, wy, resource_type.name.lower()
                    )

                if multiplier <= 0:
                    continue

                # Probabilistic spawn
                if self.rng.random() < config.spawn_rate * multiplier:
                    spawn_amount = self.rng.uniform(0.5, 2.0)
                    cell.add(resource_type, spawn_amount, config.max_per_cell * multiplier)
                    self.active_cells.add((cx, cy))
                    self.total_spawned[resource_type] += spawn_amount

    def get_resources_at(
        self,
        x: float,
        y: float
    ) -> Dict[ResourceType, float]:
        """Get all resources at a location"""
        cell_coords = self._get_cell_coords(x, y)
        cell = self.cells.get(cell_coords)
        if cell:
            return dict(cell.amounts)
        return {}

    def get_resource_at(
        self,
        x: float,
        y: float,
        resource_type: ResourceType
    ) -> float:
        """Get specific resource amount at a location"""
        cell_coords = self._get_cell_coords(x, y)
        cell = self.cells.get(cell_coords)
        if cell:
            return cell.get(resource_type)
        return 0.0

    def harvest(
        self,
        x: float,
        y: float,
        resource_type: ResourceType,
        amount: float,
        has_tool: bool = False
    ) -> float:
        """
        Harvest resources at a location.
        Returns actual amount harvested (may be less than requested).
        """
        config = self.configs.get(resource_type)
        if not config:
            return 0.0

        # Check if tool required
        if config.requires_tool and not has_tool:
            return 0.0

        cell_coords = self._get_cell_coords(x, y)
        cell = self.cells.get(cell_coords)
        if not cell:
            return 0.0

        taken = cell.take(resource_type, amount)
        self.total_harvested[resource_type] += taken

        # Set respawn timer if depleted
        if cell.get(resource_type) <= 0:
            cell.respawn_timers[resource_type] = config.respawn_time

        # Check if cell is still active
        if not any(cell.amounts.values()):
            self.active_cells.discard(cell_coords)

        return taken

    def find_nearest_resource(
        self,
        x: float,
        y: float,
        resource_type: ResourceType,
        search_radius: int = 5
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find nearest cell with resource.
        Returns (world_x, world_y, amount) or None.
        """
        center_cx, center_cy = self._get_cell_coords(x, y)

        best = None
        best_dist = float('inf')

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                cx = (center_cx + dx) % self.num_cells
                cy = (center_cy + dy) % self.num_cells

                cell = self.cells.get((cx, cy))
                if not cell:
                    continue

                amount = cell.get(resource_type)
                if amount <= 0:
                    continue

                # World coordinates
                wx = (cx + 0.5) * self.cell_size
                wy = (cy + 0.5) * self.cell_size

                dist = (x - wx) ** 2 + (y - wy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = (wx, wy, amount)

        return best

    def get_stats(self) -> Dict:
        """Get resource statistics"""
        return {
            'total_harvested': {t.name: v for t, v in self.total_harvested.items()},
            'total_spawned': {t.name: v for t, v in self.total_spawned.items()},
            'active_cells': len(self.active_cells),
        }

    def to_dict(self) -> dict:
        """Serialize for checkpointing"""
        cells_data = {}
        for coords, cell in self.cells.items():
            if cell.amounts or cell.respawn_timers:
                cells_data[f"{coords[0]},{coords[1]}"] = {
                    'amounts': {t.name: v for t, v in cell.amounts.items()},
                    'respawn_timers': {t.name: v for t, v in cell.respawn_timers.items()},
                }

        return {
            'world_size': self.world_size,
            'cell_size': self.cell_size,
            'cells': cells_data,
            'total_harvested': {t.name: v for t, v in self.total_harvested.items()},
            'total_spawned': {t.name: v for t, v in self.total_spawned.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, seed: int = 42) -> 'ResourceManager':
        """Deserialize from checkpoint"""
        manager = cls(
            world_size=data['world_size'],
            cell_size=data['cell_size'],
            seed=seed
        )

        for coords_str, cell_data in data.get('cells', {}).items():
            cx, cy = map(int, coords_str.split(','))
            cell = ResourceCell()
            for type_name, amount in cell_data.get('amounts', {}).items():
                cell.amounts[ResourceType[type_name]] = amount
            for type_name, timer in cell_data.get('respawn_timers', {}).items():
                cell.respawn_timers[ResourceType[type_name]] = timer
            manager.cells[(cx, cy)] = cell
            if any(cell.amounts.values()):
                manager.active_cells.add((cx, cy))

        for type_name, value in data.get('total_harvested', {}).items():
            manager.total_harvested[ResourceType[type_name]] = value
        for type_name, value in data.get('total_spawned', {}).items():
            manager.total_spawned[ResourceType[type_name]] = value

        return manager
