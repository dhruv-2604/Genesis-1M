"""
Simulation with LLM Integration (Phase 2)

Extends the base simulation with Tier 1 agent processing.
"""

import time
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

from .state import AgentState, WorldState, AgentArrays, FSMState, GeneIndex, AgentTier
from .events import EventLogger, EventType
from .checkpoint import CheckpointManager
from .simulation import Simulation, TickStats
from ..spatial.hash_grid import SpatialHashGrid
from ..agents.tier3 import Tier3FSM, crossover, mutate
from ..world.terrain import TerrainGenerator
from ..world.resources import ResourceManager, ResourceType
from ..inference import Tier1Processor, create_backend, PromotionScorer
from ..config import SimConfig, get_config


class LLMSimulation(Simulation):
    """
    Simulation with LLM-driven Tier 1 agents.

    Extends base simulation with:
    - Promotion scoring system
    - vLLM batch inference
    - Tier 1 action processing (1-tick delay)
    """

    def __init__(
        self,
        config: Optional[SimConfig] = None,
        checkpoint_path: Optional[str] = None,
        use_mock_llm: bool = False,
    ):
        # Initialize base simulation
        super().__init__(config=config, checkpoint_path=checkpoint_path)

        # Initialize Tier 1 processor
        self.tier1_processor = Tier1Processor(
            backend=create_backend(use_mock=use_mock_llm),
            scorer=PromotionScorer(
                global_budget=getattr(self.config, 'TIER1_BUDGET', 500),
                min_score_threshold=1.0,
                cooldown_ticks=10,
            ),
            use_mock=use_mock_llm,
        )

        self.llm_enabled = True

    def tick(self) -> TickStats:
        """Execute one tick with LLM integration"""
        tick_start = time.perf_counter()
        tick = self.world_state.tick
        births = 0
        deaths = 0

        n = self.agent_arrays.count
        alive_mask = self.agent_arrays.get_alive_mask()

        # 1. Query neighbors for all agents
        neighbor_counts, has_food, has_predator, has_mate = self._query_environment()

        # Build neighbor data for Tier 1 processing
        neighbor_data = self._build_neighbor_data()
        resource_data = self._build_resource_data()
        terrain_data = self._build_terrain_data()

        # 2. Process Tier 1 agents (applies actions from PREVIOUS tick)
        tier1_actions, promoted, demoted = [], [], []
        if self.llm_enabled:
            tier1_actions, promoted, demoted = self.tier1_processor.process_tick(
                self.agent_arrays,
                self.world_state,
                neighbor_data,
                resource_data,
                terrain_data,
            )

            # Apply Tier 1 actions
            for action in tier1_actions:
                self.tier1_processor.apply_action(
                    action, self.agent_arrays, self.world_state
                )

                # Log promotions
                if action.speech:
                    self.event_logger.log(
                        EventType.STATE_CHANGE,
                        tick,
                        agent_id=action.agent_id,
                        speech=action.speech,
                        action=action.action_type,
                    )

            # Log promotions/demotions
            for agent_id in promoted:
                self.event_logger.log_promotion(
                    tick, agent_id,
                    from_tier=3, to_tier=1,
                    reason="high_score"
                )

        # 3. Compute FSM transitions for Tier 3 agents
        new_states = self.fsm.compute_transitions(
            self.agent_arrays,
            neighbor_counts,
            has_food,
            has_predator,
            has_mate
        )

        # Only update Tier 3 agents (Tier 1 handled separately)
        for idx in self.agent_arrays.get_alive_indices():
            if self.agent_arrays.tier[idx] == AgentTier.TIER3:
                self.agent_arrays.fsm_state[idx] = new_states[idx]

        # 4. Find targets based on state
        target_x, target_y, has_target = self._compute_targets()

        # 5. Compute and apply movement
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

        # 6. Handle eating/foraging
        food_consumed = self._process_eating()

        # 7. Update energy
        self.fsm.update_energy(
            self.agent_arrays,
            food_consumed,
            self.config.BASE_ENERGY_DRAIN
        )

        # 8. Update age
        self.fsm.update_age(self.agent_arrays, self.config.MAX_AGE)

        # 9. Process reproduction
        births = self._process_reproduction()

        # 10. Process deaths
        deaths = self._process_deaths()

        # 11. Update spatial grid
        self._update_spatial_grid()

        # 12. Sync arrays back to state objects
        self.agent_arrays.sync_to_state(self.world_state)

        # 13. Regenerate resources
        self.resources.regenerate(self.terrain)

        # Increment tick
        self.world_state.tick += 1

        # Record stats
        tick_time = (time.perf_counter() - tick_start) * 1000

        tier1_count = len(self.tier1_processor.tier1_agents)
        stats = TickStats(
            tick=tick,
            population=self.world_state.population,
            births=births,
            deaths=deaths,
            tick_time_ms=tick_time,
            tier1_count=tier1_count,
            tier2_count=0,
            tier3_count=self.world_state.population - tier1_count,
        )
        self.tick_stats.append(stats)

        # Periodic logging
        if self.config.LOG_INTERVAL > 0 and tick % self.config.LOG_INTERVAL == 0:
            self._log_summary_llm(stats, len(promoted))

        # Checkpointing
        if self.config.CHECKPOINT_INTERVAL > 0 and tick % self.config.CHECKPOINT_INTERVAL == 0:
            self.save_checkpoint()

        return stats

    def _build_neighbor_data(self) -> Dict[int, List[int]]:
        """Build neighbor data for all agents"""
        neighbor_data = {}
        for idx in self.agent_arrays.get_alive_indices():
            agent_id = int(self.agent_arrays.ids[idx])
            x, y = self.agent_arrays.x[idx], self.agent_arrays.y[idx]
            neighbors = self.spatial_grid.get_neighbors(x, y, exclude_id=agent_id)
            neighbor_data[agent_id] = neighbors
        return neighbor_data

    def _build_resource_data(self) -> Dict[int, Dict[str, float]]:
        """Build resource data for all agents"""
        resource_data = {}
        for idx in self.agent_arrays.get_alive_indices():
            agent_id = int(self.agent_arrays.ids[idx])
            x, y = self.agent_arrays.x[idx], self.agent_arrays.y[idx]
            resources = self.resources.get_resources_at(x, y)
            resource_data[agent_id] = {r.name: v for r, v in resources.items()}
        return resource_data

    def _build_terrain_data(self) -> Dict[int, str]:
        """Build terrain data for all agents"""
        terrain_data = {}
        for idx in self.agent_arrays.get_alive_indices():
            agent_id = int(self.agent_arrays.ids[idx])
            x, y = self.agent_arrays.x[idx], self.agent_arrays.y[idx]
            terrain_data[agent_id] = self.terrain.get_terrain_at(x, y).name
        return terrain_data

    def _log_summary_llm(self, stats: TickStats, promotions: int) -> None:
        """Log periodic summary with LLM stats"""
        llm_stats = self.tier1_processor.get_stats()
        print(f"[Tick {stats.tick:,}] "
              f"Pop: {stats.population:,} | "
              f"T1: {stats.tier1_count} | "
              f"B: {stats.births} D: {stats.deaths} | "
              f"Promo: {promotions} | "
              f"Time: {stats.tick_time_ms:.1f}ms")

    def get_stats(self) -> Dict[str, Any]:
        """Get simulation statistics including LLM"""
        base_stats = super().get_stats()
        base_stats['tier1'] = self.tier1_processor.get_stats()
        return base_stats
