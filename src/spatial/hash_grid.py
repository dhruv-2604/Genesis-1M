"""Spatial Hash Grid for O(1) neighbor queries"""

from typing import Dict, Set, List, Tuple, Optional, Iterator
from collections import defaultdict
import numpy as np


class SpatialHashGrid:
    """
    Spatial partitioning using hash grid for efficient neighbor queries.

    World is divided into cells of size `cell_size`. Each cell maintains
    a set of agent IDs currently within it. Neighbor queries check the
    current cell plus 8 adjacent cells (3x3 neighborhood).

    For distributed execution, the grid can be "striped" horizontally,
    with each Ray actor owning a range of y-coordinates.
    """

    def __init__(
        self,
        world_size: float,
        cell_size: float,
        stripe_start: Optional[float] = None,
        stripe_end: Optional[float] = None
    ):
        self.world_size = world_size
        self.cell_size = cell_size
        self.num_cells = int(np.ceil(world_size / cell_size))

        # For distributed mode: this actor owns a horizontal stripe
        self.stripe_start = stripe_start or 0.0
        self.stripe_end = stripe_end or world_size

        # Cell storage: (cell_x, cell_y) -> set of agent IDs
        self.cells: Dict[Tuple[int, int], Set[int]] = defaultdict(set)

        # Reverse mapping: agent_id -> (cell_x, cell_y)
        self.agent_cells: Dict[int, Tuple[int, int]] = {}

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to cell coordinates"""
        cell_x = int(x / self.cell_size) % self.num_cells
        cell_y = int(y / self.cell_size) % self.num_cells
        return (cell_x, cell_y)

    def insert(self, agent_id: int, x: float, y: float) -> None:
        """Add agent to grid"""
        cell = self._get_cell(x, y)
        self.cells[cell].add(agent_id)
        self.agent_cells[agent_id] = cell

    def remove(self, agent_id: int) -> None:
        """Remove agent from grid"""
        if agent_id in self.agent_cells:
            cell = self.agent_cells.pop(agent_id)
            self.cells[cell].discard(agent_id)

    def update(self, agent_id: int, old_x: float, old_y: float, new_x: float, new_y: float) -> bool:
        """
        Update agent position. Returns True if agent crossed stripe boundary.
        """
        old_cell = self._get_cell(old_x, old_y)
        new_cell = self._get_cell(new_x, new_y)

        if old_cell != new_cell:
            self.cells[old_cell].discard(agent_id)
            self.cells[new_cell].add(agent_id)
            self.agent_cells[agent_id] = new_cell

        # Check if agent left this stripe (for distributed mode)
        crossed_boundary = (
            (old_y >= self.stripe_start and new_y < self.stripe_start) or
            (old_y < self.stripe_end and new_y >= self.stripe_end)
        )
        return crossed_boundary

    def move(self, agent_id: int, new_x: float, new_y: float) -> bool:
        """
        Move agent to new position. Returns True if crossed stripe boundary.
        """
        if agent_id not in self.agent_cells:
            self.insert(agent_id, new_x, new_y)
            return False

        old_cell = self.agent_cells[agent_id]
        new_cell = self._get_cell(new_x, new_y)

        if old_cell != new_cell:
            self.cells[old_cell].discard(agent_id)
            self.cells[new_cell].add(agent_id)
            self.agent_cells[agent_id] = new_cell

        return False

    def get_neighbors(
        self,
        x: float,
        y: float,
        exclude_id: Optional[int] = None
    ) -> List[int]:
        """
        Get all agent IDs in 3x3 cell neighborhood around (x, y).
        Excludes the agent with exclude_id if provided.
        """
        center_cell = self._get_cell(x, y)
        neighbors = []

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell_x = (center_cell[0] + dx) % self.num_cells
                cell_y = (center_cell[1] + dy) % self.num_cells
                cell = (cell_x, cell_y)

                for agent_id in self.cells.get(cell, set()):
                    if agent_id != exclude_id:
                        neighbors.append(agent_id)

        return neighbors

    def get_neighbors_in_radius(
        self,
        x: float,
        y: float,
        radius: float,
        positions: Dict[int, Tuple[float, float]],
        exclude_id: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Get agents within exact radius, with distances.
        Requires positions dict for distance calculation.
        Returns list of (agent_id, distance) tuples.
        """
        candidates = self.get_neighbors(x, y, exclude_id)
        radius_sq = radius * radius
        results = []

        for agent_id in candidates:
            if agent_id not in positions:
                continue
            ax, ay = positions[agent_id]

            # Handle world wrapping
            dx = ax - x
            dy = ay - y

            # Wrap distance for toroidal world
            if abs(dx) > self.world_size / 2:
                dx = self.world_size - abs(dx)
            if abs(dy) > self.world_size / 2:
                dy = self.world_size - abs(dy)

            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius_sq:
                results.append((agent_id, np.sqrt(dist_sq)))

        return results

    def get_cell_count(self, cell: Tuple[int, int]) -> int:
        """Get number of agents in a specific cell"""
        return len(self.cells.get(cell, set()))

    def get_density_map(self) -> np.ndarray:
        """Return 2D array of agent counts per cell"""
        density = np.zeros((self.num_cells, self.num_cells), dtype=np.int32)
        for (cx, cy), agents in self.cells.items():
            density[cx, cy] = len(agents)
        return density

    def clear(self) -> None:
        """Remove all agents"""
        self.cells.clear()
        self.agent_cells.clear()

    def rebuild(self, positions: Dict[int, Tuple[float, float]]) -> None:
        """Rebuild grid from scratch with given positions"""
        self.clear()
        for agent_id, (x, y) in positions.items():
            self.insert(agent_id, x, y)

    def __len__(self) -> int:
        return len(self.agent_cells)

    def __contains__(self, agent_id: int) -> bool:
        return agent_id in self.agent_cells


class SpatialHashGridVectorized:
    """
    Vectorized spatial hash grid using NumPy arrays.
    More efficient for batch updates of many agents.
    """

    def __init__(self, world_size: float, cell_size: float, max_agents: int = 1_100_000):
        self.world_size = world_size
        self.cell_size = cell_size
        self.num_cells = int(np.ceil(world_size / cell_size))
        self.max_agents = max_agents

        # Pre-allocated arrays for sorting approach
        self.cell_indices = np.zeros(max_agents, dtype=np.int32)
        self.agent_order = np.zeros(max_agents, dtype=np.int32)

        # Cell boundaries after sorting
        self.cell_starts = np.zeros(self.num_cells * self.num_cells + 1, dtype=np.int32)

    def compute_cell_index(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute 1D cell index from x, y coordinates"""
        cell_x = (x / self.cell_size).astype(np.int32) % self.num_cells
        cell_y = (y / self.cell_size).astype(np.int32) % self.num_cells
        return cell_y * self.num_cells + cell_x

    def build(
        self,
        agent_ids: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        alive: np.ndarray
    ) -> None:
        """
        Build spatial index from arrays.
        After calling, use query methods to find neighbors.
        """
        n = len(agent_ids)
        if n == 0:
            return

        # Compute cell for each agent
        alive_mask = alive[:n]
        cell_idx = self.compute_cell_index(x[:n], y[:n])

        # Set dead agents to invalid cell
        cell_idx[~alive_mask] = self.num_cells * self.num_cells

        # Sort agents by cell index
        order = np.argsort(cell_idx)
        sorted_cells = cell_idx[order]

        # Store sorted order
        self.agent_order[:n] = agent_ids[order]
        self.cell_indices[:n] = sorted_cells

        # Compute cell start positions
        self.cell_starts.fill(n)  # Default to end

        # Find where each cell starts
        changes = np.where(np.diff(sorted_cells, prepend=-1) != 0)[0]
        for i, pos in enumerate(changes):
            if sorted_cells[pos] < self.num_cells * self.num_cells:
                self.cell_starts[sorted_cells[pos]] = pos

        # Fill in gaps (cells with no agents point to next cell's start)
        for i in range(self.num_cells * self.num_cells - 1, -1, -1):
            if self.cell_starts[i] == n:
                self.cell_starts[i] = self.cell_starts[i + 1] if i + 1 < len(self.cell_starts) else n

    def get_neighbors_batch(
        self,
        query_x: np.ndarray,
        query_y: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        radius: float,
        max_neighbors: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find neighbors for multiple query points.
        Returns (neighbor_ids, neighbor_counts) arrays.
        """
        n_queries = len(query_x)
        result_ids = np.full((n_queries, max_neighbors), -1, dtype=np.int32)
        result_counts = np.zeros(n_queries, dtype=np.int32)

        radius_sq = radius * radius

        for i in range(n_queries):
            qx, qy = query_x[i], query_y[i]
            center_cx = int(qx / self.cell_size) % self.num_cells
            center_cy = int(qy / self.cell_size) % self.num_cells

            count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cx = (center_cx + dx) % self.num_cells
                    cy = (center_cy + dy) % self.num_cells
                    cell_idx = cy * self.num_cells + cx

                    start = self.cell_starts[cell_idx]
                    end = self.cell_starts[cell_idx + 1] if cell_idx + 1 < len(self.cell_starts) else len(self.agent_order)

                    for j in range(start, end):
                        if count >= max_neighbors:
                            break
                        aid = self.agent_order[j]

                        # Distance check with wrapping
                        ddx = x[j] - qx
                        ddy = y[j] - qy
                        if abs(ddx) > self.world_size / 2:
                            ddx = self.world_size - abs(ddx)
                        if abs(ddy) > self.world_size / 2:
                            ddy = self.world_size - abs(ddy)

                        if ddx * ddx + ddy * ddy <= radius_sq:
                            result_ids[i, count] = aid
                            count += 1

                    if count >= max_neighbors:
                        break
                if count >= max_neighbors:
                    break

            result_counts[i] = count

        return result_ids, result_counts
