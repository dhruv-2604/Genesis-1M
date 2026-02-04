"""
Optimized simulation tick - minimizes Python loops in hot path.

This version batches operations more aggressively for better performance.
"""

import time
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

from .state import AgentState, WorldState, AgentArrays, FSMState, GeneIndex, AgentTier
from .events import EventLogger
from .checkpoint import CheckpointManager
from ..spatial.hash_grid import SpatialHashGrid
from ..agents.tier3 import Tier3FSM, crossover, mutate
from ..world.terrain import TerrainGenerator
from ..world.resources import ResourceManager, ResourceType
from ..config import SimConfig, get_config


class FastSimulation:
    """
    Performance-optimized simulation with minimal Python loops.

    Key optimizations:
    - Batched neighbor queries using numpy
    - Vectorized FSM state updates
    - Reduced per-agent iteration
    - Sampling-based food detection
    """

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        checkpoint_path: Optional[str] = None
    ):
        self.config = config or get_config()
        self.rng = np.random.default_rng(self.config.SEED)

        # Core state
        self.world_state = WorldState(world_size=self.config.WORLD_SIZE)
        self.arrays = AgentArrays(max_agents=self.config.MAX_AGENTS)

        # Subsystems
        self.grid = SpatialHashGrid(
            world_size=self.config.WORLD_SIZE,
            cell_size=self.config.CELL_SIZE
        )
        self.fsm = Tier3FSM(
            world_size=self.config.WORLD_SIZE,
            cell_size=self.config.CELL_SIZE
        )
        self.terrain = TerrainGenerator(
            world_size=self.config.WORLD_SIZE,
            seed=self.config.SEED
        )
        self.resources = ResourceManager(
            world_size=self.config.WORLD_SIZE,
            cell_size=self.config.CELL_SIZE,
            seed=self.config.SEED
        )

        # Logging
        self.event_logger = EventLogger(
            log_dir=self.config.EVENT_LOG_DIR,
            buffer_size=50000
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.CHECKPOINT_DIR,
            max_checkpoints=self.config.MAX_CHECKPOINTS
        )

        # Pre-allocated work arrays
        self.n_max = self.config.MAX_AGENTS
        self._neighbor_counts = np.zeros(self.n_max, dtype=np.int32)
        self._has_food = np.zeros(self.n_max, dtype=np.bool_)
        self._has_predator = np.zeros(self.n_max, dtype=np.bool_)
        self._has_mate = np.zeros(self.n_max, dtype=np.bool_)
        self._target_x = np.zeros(self.n_max, dtype=np.float32)
        self._target_y = np.zeros(self.n_max, dtype=np.float32)
        self._has_target = np.zeros(self.n_max, dtype=np.bool_)
        self._food_consumed = np.zeros(self.n_max, dtype=np.float32)

        self.running = False

        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        else:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize world"""
        self.resources.initialize_resources(self.terrain, 0.3)

        # Spawn agents
        for _ in range(self.config.INITIAL_AGENT_COUNT):
            agent_id = self.world_state.get_new_id()
            x = self.rng.uniform(0, self.config.WORLD_SIZE)
            y = self.rng.uniform(0, self.config.WORLD_SIZE)

            agent = AgentState(
                id=agent_id, x=x, y=y,
                energy=self.config.STARTING_ENERGY,
                genes=self.rng.uniform(0.5, 1.5, GeneIndex.NUM_GENES).astype(np.float32)
            )
            self.world_state.add_agent(agent)

        self.arrays.sync_from_state(self.world_state)
        self._rebuild_grid()

    def _rebuild_grid(self) -> None:
        """Rebuild spatial grid from current positions"""
        self.grid.clear()
        n = self.arrays.count
        for i in range(n):
            if self.arrays.alive[i]:
                self.grid.insert(
                    int(self.arrays.ids[i]),
                    float(self.arrays.x[i]),
                    float(self.arrays.y[i])
                )

    def tick(self) -> Dict[str, Any]:
        """Execute one tick with optimized batching"""
        tick_start = time.perf_counter()
        tick = self.world_state.tick
        n = self.arrays.count

        # Reset work arrays
        self._neighbor_counts[:n] = 0
        self._has_food[:n] = False
        self._has_mate[:n] = False
        self._has_target[:n] = False
        self._food_consumed[:n] = 0

        alive_mask = self.arrays.get_alive_mask()
        alive_idx = np.where(alive_mask)[0]

        # Vectorized: check food availability using sampling
        # Sample 10% of alive agents for expensive food checks
        sample_size = max(1, len(alive_idx) // 10)
        sample_idx = self.rng.choice(alive_idx, size=min(sample_size, len(alive_idx)), replace=False)

        for idx in sample_idx:
            x, y = self.arrays.x[idx], self.arrays.y[idx]
            resources = self.resources.get_resources_at(x, y)
            if any(v > 0 for v in resources.values()):
                # Propagate to neighbors in same cell
                cx = int(x / self.config.CELL_SIZE) % self.grid.num_cells
                cy = int(y / self.config.CELL_SIZE) % self.grid.num_cells
                cell_agents = self.grid.cells.get((cx, cy), set())
                for aid in cell_agents:
                    if aid in self.arrays.id_to_index:
                        self._has_food[self.arrays.id_to_index[aid]] = True

        # Simplified neighbor counting using cell density
        for idx in alive_idx:
            x, y = self.arrays.x[idx], self.arrays.y[idx]
            cx = int(x / self.config.CELL_SIZE) % self.grid.num_cells
            cy = int(y / self.config.CELL_SIZE) % self.grid.num_cells

            count = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell = ((cx + dx) % self.grid.num_cells, (cy + dy) % self.grid.num_cells)
                    count += len(self.grid.cells.get(cell, set()))

            self._neighbor_counts[idx] = count - 1  # Exclude self

            # Check for mates if enough neighbors
            if count > 1 and self.arrays.energy[idx] >= self.config.REPRODUCTION_ENERGY_THRESHOLD:
                self._has_mate[idx] = True

        # FSM transitions (fully vectorized)
        new_states = self.fsm.compute_transitions(
            self.arrays,
            self._neighbor_counts[:n],
            self._has_food[:n],
            self._has_predator[:n],
            self._has_mate[:n]
        )
        self.arrays.fsm_state[:n] = new_states

        # Vectorized movement
        dx, dy = self.fsm.compute_movement(
            self.arrays,
            self._target_x[:n],
            self._target_y[:n],
            self._has_target[:n],
            self.rng
        )
        self.fsm.apply_movement(self.arrays, dx, dy)

        # Batch eating for hungry agents
        hungry_mask = alive_mask & (self.arrays.fsm_state[:n] == FSMState.SEEK_FOOD)
        hungry_idx = np.where(hungry_mask)[0]

        for idx in hungry_idx:
            x, y = self.arrays.x[idx], self.arrays.y[idx]
            harvested = self.resources.harvest(x, y, ResourceType.BERRIES, 1.0)
            if harvested > 0:
                self._food_consumed[idx] = harvested * 20.0

        # Update energy and age
        self.fsm.update_energy(self.arrays, self._food_consumed[:n], self.config.BASE_ENERGY_DRAIN)
        self.fsm.update_age(self.arrays, self.config.MAX_AGE)

        # Process deaths
        deaths = 0
        for idx in alive_idx:
            if self.arrays.energy[idx] <= 0.5 or self.arrays.age[idx] >= self.config.MAX_AGE:
                agent_id = int(self.arrays.ids[idx])
                self.arrays.remove_agent(agent_id)
                self.grid.remove(agent_id)
                self.world_state.remove_agent(agent_id)
                self.world_state.total_deaths += 1
                deaths += 1

        # Process reproduction (sample-based for performance)
        births = 0
        if len(alive_idx) > 1 and self.world_state.population < self.config.MAX_AGENTS:
            # Check subset of agents for reproduction
            mate_idx = np.where(self._has_mate[:n] & alive_mask)[0]
            if len(mate_idx) >= 2:
                # Randomly pair some agents
                pairs_to_check = min(10, len(mate_idx) // 2)
                self.rng.shuffle(mate_idx)

                for i in range(pairs_to_check):
                    if i * 2 + 1 >= len(mate_idx):
                        break

                    idx1, idx2 = mate_idx[i * 2], mate_idx[i * 2 + 1]

                    # Distance check
                    dist = np.sqrt(
                        (self.arrays.x[idx1] - self.arrays.x[idx2])**2 +
                        (self.arrays.y[idx1] - self.arrays.y[idx2])**2
                    )

                    if dist < self.config.INTERACTION_RANGE * 2:
                        # Fertility check
                        fertility = (
                            self.arrays.genes[idx1, GeneIndex.FERTILITY] *
                            self.arrays.genes[idx2, GeneIndex.FERTILITY]
                        )

                        if self.rng.random() < fertility:
                            # Create child
                            child_genes = crossover(
                                self.arrays.genes[idx1],
                                self.arrays.genes[idx2],
                                self.rng
                            )
                            child_genes = mutate(child_genes, self.rng)

                            child_id = self.world_state.get_new_id()
                            child = AgentState(
                                id=child_id,
                                x=(self.arrays.x[idx1] + self.arrays.x[idx2]) / 2,
                                y=(self.arrays.y[idx1] + self.arrays.y[idx2]) / 2,
                                energy=self.config.CHILD_STARTING_ENERGY,
                                genes=child_genes,
                                parent_ids=(int(self.arrays.ids[idx1]), int(self.arrays.ids[idx2]))
                            )

                            self.world_state.add_agent(child)
                            self.arrays.add_agent(child)
                            self.grid.insert(child.id, child.x, child.y)

                            # Deplete parent energy
                            self.arrays.energy[idx1] -= self.config.REPRODUCTION_ENERGY_COST
                            self.arrays.energy[idx2] -= self.config.REPRODUCTION_ENERGY_COST
                            self.arrays.reproductive_cooldown[idx1] = self.config.REPRODUCTION_COOLDOWN
                            self.arrays.reproductive_cooldown[idx2] = self.config.REPRODUCTION_COOLDOWN

                            self.world_state.total_births += 1
                            births += 1

        # Update grid positions
        for idx in self.arrays.get_alive_indices():
            self.grid.move(int(self.arrays.ids[idx]), float(self.arrays.x[idx]), float(self.arrays.y[idx]))

        # Regenerate resources periodically
        if tick % 10 == 0:
            self.resources.regenerate(self.terrain)

        self.world_state.tick += 1

        tick_time = (time.perf_counter() - tick_start) * 1000

        return {
            'tick': tick,
            'population': self.world_state.population,
            'births': births,
            'deaths': deaths,
            'tick_time_ms': tick_time,
        }

    def run(self, num_ticks: int) -> None:
        """Run for N ticks"""
        self.running = True
        for _ in range(num_ticks):
            if not self.running:
                break
            stats = self.tick()
            if self.config.LOG_INTERVAL > 0 and stats['tick'] % self.config.LOG_INTERVAL == 0:
                print(f"[Tick {stats['tick']:,}] Pop: {stats['population']:,} | "
                      f"B: {stats['births']} D: {stats['deaths']} | {stats['tick_time_ms']:.1f}ms")

    def get_stats(self) -> Dict[str, Any]:
        return {
            'tick': self.world_state.tick,
            'population': self.world_state.population,
            'total_births': self.world_state.total_births,
            'total_deaths': self.world_state.total_deaths,
        }

    def _load_checkpoint(self, path: str) -> None:
        data = self.checkpoint_manager.load(path)
        self.world_state = data['world_state']
        self.arrays.sync_from_state(self.world_state)
        self._rebuild_grid()
