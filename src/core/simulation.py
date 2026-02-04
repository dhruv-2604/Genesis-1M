"""Main Simulation Engine - Tick Loop and Orchestration"""

import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .state import AgentState, WorldState, AgentArrays, FSMState, GeneIndex, AgentTier
from .events import EventLogger, EventType
from .checkpoint import CheckpointManager
from ..spatial.hash_grid import SpatialHashGrid
from ..agents.tier3 import Tier3FSM, crossover, mutate, create_child
from ..world.terrain import TerrainGenerator
from ..world.resources import ResourceManager, ResourceType
from ..config import SimConfig, get_config


@dataclass
class TickStats:
    """Statistics for a single tick"""
    tick: int
    population: int
    births: int
    deaths: int
    tick_time_ms: float
    tier1_count: int
    tier2_count: int
    tier3_count: int


class Simulation:
    """
    Main simulation orchestrator.

    Manages the tick loop, coordinates all subsystems, and handles
    the promotion/demotion of agents between tiers.
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
        self.agent_arrays = AgentArrays(max_agents=self.config.MAX_AGENTS)

        # Subsystems
        self.spatial_grid = SpatialHashGrid(
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

        # Logging and checkpointing
        self.event_logger = EventLogger(
            log_dir=self.config.EVENT_LOG_DIR,
            buffer_size=10000
        )
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=self.config.CHECKPOINT_DIR,
            max_checkpoints=self.config.MAX_CHECKPOINTS
        )

        # Performance tracking
        self.tick_stats: List[TickStats] = []
        self.running = False

        # Load checkpoint or initialize
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            self._initialize_world()

    def _initialize_world(self) -> None:
        """Initialize world with starting agents and resources"""
        # Initialize resources
        self.resources.initialize_resources(
            terrain_generator=self.terrain,
            initial_density=0.3
        )

        # Spawn initial agents
        for _ in range(self.config.INITIAL_AGENT_COUNT):
            self._spawn_agent()

        # Sync to arrays for vectorized operations
        self.agent_arrays.sync_from_state(self.world_state)

        # Build spatial grid
        positions = {
            aid: (a.x, a.y)
            for aid, a in self.world_state.agents.items()
        }
        self.spatial_grid.rebuild(positions)

    def _spawn_agent(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        genes: Optional[np.ndarray] = None,
        parent_ids: Tuple[int, int] = (-1, -1),
        energy: Optional[float] = None
    ) -> AgentState:
        """Spawn a new agent"""
        agent_id = self.world_state.get_new_id()

        # Random position if not specified
        if x is None:
            x = self.rng.uniform(0, self.config.WORLD_SIZE)
        if y is None:
            y = self.rng.uniform(0, self.config.WORLD_SIZE)

        # Ensure position is passable
        attempts = 0
        while not self.terrain.is_passable(x, y) and attempts < 10:
            x = self.rng.uniform(0, self.config.WORLD_SIZE)
            y = self.rng.uniform(0, self.config.WORLD_SIZE)
            attempts += 1

        # Random genes if not specified
        if genes is None:
            genes = self.rng.uniform(0.5, 1.5, GeneIndex.NUM_GENES).astype(np.float32)

        agent = AgentState(
            id=agent_id,
            x=x,
            y=y,
            energy=energy or self.config.STARTING_ENERGY,
            genes=genes,
            parent_ids=parent_ids,
        )

        self.world_state.add_agent(agent)
        return agent

    def tick(self) -> TickStats:
        """Execute one simulation tick"""
        tick_start = time.perf_counter()
        tick = self.world_state.tick
        births = 0
        deaths = 0

        n = self.agent_arrays.count
        alive_mask = self.agent_arrays.get_alive_mask()

        # 1. Query neighbors for all agents
        neighbor_counts, has_food, has_predator, has_mate = self._query_environment()

        # 2. Compute FSM transitions
        new_states = self.fsm.compute_transitions(
            self.agent_arrays,
            neighbor_counts,
            has_food,
            has_predator,
            has_mate
        )
        self.agent_arrays.fsm_state[:n] = new_states

        # 3. Find targets based on state
        target_x, target_y, has_target = self._compute_targets()

        # 4. Compute and apply movement
        dx, dy = self.fsm.compute_movement(
            self.agent_arrays,
            target_x,
            target_y,
            has_target,
            self.rng
        )

        # Apply terrain speed modifiers
        for i in range(n):
            if alive_mask[i]:
                mult = self.terrain.get_movement_multiplier(
                    self.agent_arrays.x[i],
                    self.agent_arrays.y[i]
                )
                dx[i] *= mult
                dy[i] *= mult

        self.fsm.apply_movement(self.agent_arrays, dx, dy)

        # 5. Handle eating/foraging
        food_consumed = self._process_eating()

        # 6. Update energy
        self.fsm.update_energy(
            self.agent_arrays,
            food_consumed,
            self.config.BASE_ENERGY_DRAIN
        )

        # 7. Update age
        self.fsm.update_age(self.agent_arrays, self.config.MAX_AGE)

        # 8. Process reproduction
        births = self._process_reproduction()

        # 9. Process deaths
        deaths = self._process_deaths()

        # 10. Update spatial grid
        self._update_spatial_grid()

        # 11. Sync arrays back to state objects
        self.agent_arrays.sync_to_state(self.world_state)

        # 12. Regenerate resources
        self.resources.regenerate(self.terrain)

        # Increment tick
        self.world_state.tick += 1

        # Record stats
        tick_time = (time.perf_counter() - tick_start) * 1000
        stats = TickStats(
            tick=tick,
            population=self.world_state.population,
            births=births,
            deaths=deaths,
            tick_time_ms=tick_time,
            tier1_count=0,  # Phase 2
            tier2_count=0,  # Phase 2
            tier3_count=self.world_state.population,
        )
        self.tick_stats.append(stats)

        # Periodic logging
        if self.config.LOG_INTERVAL > 0 and tick % self.config.LOG_INTERVAL == 0:
            self._log_summary(stats)

        # Checkpointing
        if self.config.CHECKPOINT_INTERVAL > 0 and tick % self.config.CHECKPOINT_INTERVAL == 0:
            self.save_checkpoint()

        return stats

    def _query_environment(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Query spatial grid for environmental information"""
        n = self.agent_arrays.count
        neighbor_counts = np.zeros(n, dtype=np.int32)
        has_food = np.zeros(n, dtype=np.bool_)
        has_predator = np.zeros(n, dtype=np.bool_)
        has_mate = np.zeros(n, dtype=np.bool_)

        alive_indices = self.agent_arrays.get_alive_indices()

        for idx in alive_indices:
            agent_id = self.agent_arrays.ids[idx]
            x = self.agent_arrays.x[idx]
            y = self.agent_arrays.y[idx]

            # Get neighbors
            neighbor_ids = self.spatial_grid.get_neighbors(x, y, exclude_id=agent_id)
            neighbor_counts[idx] = len(neighbor_ids)

            # Check for food
            resources = self.resources.get_resources_at(x, y)
            has_food[idx] = any(
                v > 0 for k, v in resources.items()
                if k in [ResourceType.BERRIES, ResourceType.MEAT, ResourceType.FISH]
            )

            # Check for potential mates (other agents with sufficient energy)
            for nid in neighbor_ids[:10]:  # Limit check
                if nid in self.agent_arrays.id_to_index:
                    nidx = self.agent_arrays.id_to_index[nid]
                    if (self.agent_arrays.alive[nidx] and
                        self.agent_arrays.energy[nidx] >= self.config.REPRODUCTION_ENERGY_THRESHOLD and
                        self.agent_arrays.age[nidx] >= self.config.MATURITY_AGE):
                        has_mate[idx] = True
                        break

        return neighbor_counts, has_food, has_predator, has_mate

    def _compute_targets(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute movement targets based on FSM state"""
        n = self.agent_arrays.count
        target_x = np.zeros(n, dtype=np.float32)
        target_y = np.zeros(n, dtype=np.float32)
        has_target = np.zeros(n, dtype=np.bool_)

        alive_indices = self.agent_arrays.get_alive_indices()

        for idx in alive_indices:
            state = FSMState(self.agent_arrays.fsm_state[idx])
            x = self.agent_arrays.x[idx]
            y = self.agent_arrays.y[idx]

            if state == FSMState.SEEK_FOOD:
                # Find nearest food
                food = self.resources.find_nearest_resource(
                    x, y, ResourceType.BERRIES, search_radius=3
                )
                if food:
                    target_x[idx], target_y[idx], _ = food
                    has_target[idx] = True

            elif state == FSMState.SEEK_MATE:
                # Find nearest potential mate
                agent_id = self.agent_arrays.ids[idx]
                neighbor_ids = self.spatial_grid.get_neighbors(x, y, exclude_id=agent_id)

                for nid in neighbor_ids[:5]:
                    if nid in self.agent_arrays.id_to_index:
                        nidx = self.agent_arrays.id_to_index[nid]
                        if (self.agent_arrays.alive[nidx] and
                            self.agent_arrays.energy[nidx] >= self.config.REPRODUCTION_ENERGY_THRESHOLD):
                            target_x[idx] = self.agent_arrays.x[nidx]
                            target_y[idx] = self.agent_arrays.y[nidx]
                            has_target[idx] = True
                            break

        return target_x, target_y, has_target

    def _process_eating(self) -> np.ndarray:
        """Process foraging and eating"""
        n = self.agent_arrays.count
        food_consumed = np.zeros(n, dtype=np.float32)

        alive_indices = self.agent_arrays.get_alive_indices()

        for idx in alive_indices:
            state = FSMState(self.agent_arrays.fsm_state[idx])
            if state not in [FSMState.SEEK_FOOD, FSMState.WANDER]:
                continue

            x = self.agent_arrays.x[idx]
            y = self.agent_arrays.y[idx]

            # Try to harvest berries (easiest)
            harvested = self.resources.harvest(
                x, y, ResourceType.BERRIES, amount=1.0
            )
            if harvested > 0:
                energy_gain = harvested * 20.0  # Berry energy value
                food_consumed[idx] = energy_gain

                # Log eating event
                self.event_logger.log_eat(
                    tick=self.world_state.tick,
                    agent_id=int(self.agent_arrays.ids[idx]),
                    resource_type='berries',
                    amount=energy_gain,
                    x=x,
                    y=y
                )

        return food_consumed

    def _process_reproduction(self) -> int:
        """Process reproduction between nearby agents"""
        births = 0
        tick = self.world_state.tick

        # Find pairs of agents close enough to reproduce
        potential_pairs = []
        checked = set()

        alive_indices = self.agent_arrays.get_alive_indices()

        for idx in alive_indices:
            agent_id = self.agent_arrays.ids[idx]
            if agent_id in checked:
                continue

            # Check reproduction conditions
            if not self._can_reproduce(idx):
                continue

            x = self.agent_arrays.x[idx]
            y = self.agent_arrays.y[idx]

            # Find nearby mate
            neighbor_ids = self.spatial_grid.get_neighbors(x, y, exclude_id=agent_id)

            for nid in neighbor_ids:
                if nid in checked:
                    continue
                if nid not in self.agent_arrays.id_to_index:
                    continue

                nidx = self.agent_arrays.id_to_index[nid]
                if self._can_reproduce(nidx):
                    # Check proximity
                    nx, ny = self.agent_arrays.x[nidx], self.agent_arrays.y[nidx]
                    dist = np.sqrt((x - nx)**2 + (y - ny)**2)

                    if dist < self.config.INTERACTION_RANGE:
                        potential_pairs.append((agent_id, nid))
                        checked.add(agent_id)
                        checked.add(nid)
                        break

        # Process pairs
        for parent1_id, parent2_id in potential_pairs:
            if self.world_state.population >= self.config.MAX_AGENTS:
                break

            idx1 = self.agent_arrays.id_to_index[parent1_id]
            idx2 = self.agent_arrays.id_to_index[parent2_id]

            # Fertility check
            fertility = (
                self.agent_arrays.genes[idx1, GeneIndex.FERTILITY] *
                self.agent_arrays.genes[idx2, GeneIndex.FERTILITY]
            )
            if self.rng.random() > fertility:
                continue

            # Create child
            parent1_genes = self.agent_arrays.genes[idx1]
            parent2_genes = self.agent_arrays.genes[idx2]
            child_genes = crossover(parent1_genes, parent2_genes, self.rng)
            child_genes = mutate(child_genes, self.rng)

            child_x = (self.agent_arrays.x[idx1] + self.agent_arrays.x[idx2]) / 2
            child_y = (self.agent_arrays.y[idx1] + self.agent_arrays.y[idx2]) / 2

            child = self._spawn_agent(
                x=child_x,
                y=child_y,
                genes=child_genes,
                parent_ids=(parent1_id, parent2_id),
                energy=self.config.CHILD_STARTING_ENERGY
            )

            # Add to arrays
            self.agent_arrays.add_agent(child)
            self.spatial_grid.insert(child.id, child.x, child.y)

            # Deplete parent energy
            self.agent_arrays.energy[idx1] -= self.config.REPRODUCTION_ENERGY_COST
            self.agent_arrays.energy[idx2] -= self.config.REPRODUCTION_ENERGY_COST

            # Set cooldown
            self.agent_arrays.reproductive_cooldown[idx1] = self.config.REPRODUCTION_COOLDOWN
            self.agent_arrays.reproductive_cooldown[idx2] = self.config.REPRODUCTION_COOLDOWN

            # Log birth
            self.event_logger.log_birth(
                tick=tick,
                child_id=child.id,
                parent_ids=(parent1_id, parent2_id),
                x=child.x,
                y=child.y,
                genes=child_genes.tolist()
            )

            self.world_state.total_births += 1
            births += 1

        return births

    def _can_reproduce(self, idx: int) -> bool:
        """Check if agent at index can reproduce"""
        return (
            self.agent_arrays.alive[idx] and
            self.agent_arrays.energy[idx] >= self.config.REPRODUCTION_ENERGY_THRESHOLD and
            self.agent_arrays.age[idx] >= self.config.MATURITY_AGE and
            self.agent_arrays.reproductive_cooldown[idx] == 0
        )

    def _process_deaths(self) -> int:
        """Process agent deaths"""
        deaths = 0
        tick = self.world_state.tick

        alive_indices = self.agent_arrays.get_alive_indices()

        for idx in alive_indices:
            if not self.agent_arrays.alive[idx]:
                continue

            agent_id = self.agent_arrays.ids[idx]
            should_die = False
            cause = None

            # Death by starvation (use small threshold for floating point)
            if self.agent_arrays.energy[idx] <= 0.5:
                should_die = True
                cause = 'starvation'

            # Death by old age
            elif self.agent_arrays.age[idx] >= self.config.MAX_AGE:
                should_die = True
                cause = 'old_age'

            if should_die:
                x = float(self.agent_arrays.x[idx])
                y = float(self.agent_arrays.y[idx])
                age = int(self.agent_arrays.age[idx])

                # Log death
                self.event_logger.log_death(
                    tick=tick,
                    agent_id=int(agent_id),
                    cause=cause,
                    age=age,
                    x=x,
                    y=y
                )

                # Remove from systems
                self.agent_arrays.remove_agent(agent_id)
                self.spatial_grid.remove(agent_id)
                self.world_state.remove_agent(agent_id)

                self.world_state.total_deaths += 1
                deaths += 1

        return deaths

    def _update_spatial_grid(self) -> None:
        """Update spatial grid with new positions"""
        alive_indices = self.agent_arrays.get_alive_indices()

        for idx in alive_indices:
            agent_id = self.agent_arrays.ids[idx]
            x = self.agent_arrays.x[idx]
            y = self.agent_arrays.y[idx]
            self.spatial_grid.move(agent_id, x, y)

    def _log_summary(self, stats: TickStats) -> None:
        """Log periodic summary"""
        print(f"[Tick {stats.tick:,}] "
              f"Pop: {stats.population:,} | "
              f"Births: {stats.births} | "
              f"Deaths: {stats.deaths} | "
              f"Time: {stats.tick_time_ms:.1f}ms")

    def run(self, num_ticks: int, target_tps: Optional[float] = None) -> None:
        """Run simulation for N ticks"""
        self.running = True
        target_tps = target_tps or self.config.TICK_RATE_TARGET

        tick_interval = 1.0 / target_tps if target_tps > 0 else 0

        for i in range(num_ticks):
            if not self.running:
                break

            tick_start = time.perf_counter()
            self.tick()
            tick_elapsed = time.perf_counter() - tick_start

            # Rate limiting
            if tick_interval > tick_elapsed:
                time.sleep(tick_interval - tick_elapsed)

        self.running = False

    def stop(self) -> None:
        """Stop the simulation"""
        self.running = False

    def save_checkpoint(self, metadata: Optional[Dict] = None) -> str:
        """Save current state to checkpoint"""
        # Sync arrays to state
        self.agent_arrays.sync_to_state(self.world_state)

        checkpoint_id = self.checkpoint_manager.save(
            world_state=self.world_state,
            resource_state=self.resources.to_dict(),
            metadata=metadata or {}
        )

        print(f"Checkpoint saved: {checkpoint_id}")
        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: Optional[str] = None) -> None:
        """Load state from checkpoint"""
        data = self.checkpoint_manager.load(checkpoint_id)

        self.world_state = data['world_state']

        if data.get('resource_state'):
            self.resources = ResourceManager.from_dict(
                data['resource_state'],
                seed=self.config.SEED
            )

        # Rebuild arrays and spatial grid
        self.agent_arrays = AgentArrays(max_agents=self.config.MAX_AGENTS)
        self.agent_arrays.sync_from_state(self.world_state)

        positions = {
            aid: (a.x, a.y)
            for aid, a in self.world_state.agents.items()
        }
        self.spatial_grid.rebuild(positions)

        print(f"Loaded checkpoint: {data['checkpoint_id']} (tick {data['tick']})")

    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        return {
            'tick': self.world_state.tick,
            'population': self.world_state.population,
            'total_births': self.world_state.total_births,
            'total_deaths': self.world_state.total_deaths,
            'resources': self.resources.get_stats(),
            'events': self.event_logger.get_stats(),
        }

    def shutdown(self) -> None:
        """Clean shutdown"""
        self.running = False
        self.event_logger.close()
        self.save_checkpoint(metadata={'shutdown': True})
