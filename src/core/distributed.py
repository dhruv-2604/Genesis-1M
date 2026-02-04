"""
Distributed simulation using Ray for multi-GPU scaling.

Each StripeActor owns a horizontal stripe of the world and manages
agents within that stripe. Agents crossing stripe boundaries trigger
inter-actor communication.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Ray import is optional for Phase 1
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .state import AgentState, WorldState, AgentArrays, FSMState
from ..spatial.hash_grid import SpatialHashGrid
from ..agents.tier3 import Tier3FSM
from ..config import SimConfig


if RAY_AVAILABLE:
    @ray.remote
    class StripeActor:
        """
        Ray actor managing a horizontal stripe of the world.

        Each actor:
        - Owns agents within its y-coordinate range
        - Maintains local spatial grid
        - Processes FSM updates for its agents
        - Handles agents entering/leaving its stripe
        """

        def __init__(
            self,
            actor_id: int,
            stripe_start: float,
            stripe_end: float,
            config: SimConfig
        ):
            self.actor_id = actor_id
            self.stripe_start = stripe_start
            self.stripe_end = stripe_end
            self.config = config

            self.rng = np.random.default_rng(config.SEED + actor_id)

            # Local state
            self.agents: Dict[int, AgentState] = {}
            self.arrays = AgentArrays(max_agents=config.MAX_AGENTS // 8 + 10000)

            # Spatial grid for this stripe
            self.grid = SpatialHashGrid(
                world_size=config.WORLD_SIZE,
                cell_size=config.CELL_SIZE,
                stripe_start=stripe_start,
                stripe_end=stripe_end
            )

            # FSM processor
            self.fsm = Tier3FSM(
                world_size=config.WORLD_SIZE,
                cell_size=config.CELL_SIZE
            )

            # Pending transfers
            self.outgoing_agents: List[AgentState] = []
            self.incoming_agents: List[AgentState] = []

        def add_agent(self, agent: AgentState) -> None:
            """Add agent to this stripe"""
            self.agents[agent.id] = agent
            self.arrays.add_agent(agent)
            self.grid.insert(agent.id, agent.x, agent.y)

        def receive_agents(self, agents: List[dict]) -> None:
            """Receive agents transferred from other stripes"""
            for agent_data in agents:
                agent = AgentState.from_dict(agent_data)
                self.add_agent(agent)

        def get_outgoing_agents(self) -> List[dict]:
            """Get agents that need to transfer to other stripes"""
            result = [a.to_dict() for a in self.outgoing_agents]
            self.outgoing_agents.clear()
            return result

        def tick(self, tick_num: int) -> Dict[str, Any]:
            """Execute one tick for agents in this stripe"""
            if not self.agents:
                return {'births': 0, 'deaths': 0, 'transfers': 0}

            n = self.arrays.count
            alive_mask = self.arrays.get_alive_mask()

            # Query neighbors
            neighbor_counts = np.zeros(n, dtype=np.int32)
            has_food = np.ones(n, dtype=np.bool_)  # Simplified for distributed
            has_predator = np.zeros(n, dtype=np.bool_)
            has_mate = np.zeros(n, dtype=np.bool_)

            for idx in self.arrays.get_alive_indices():
                agent_id = self.arrays.ids[idx]
                x, y = self.arrays.x[idx], self.arrays.y[idx]
                neighbors = self.grid.get_neighbors(x, y, exclude_id=agent_id)
                neighbor_counts[idx] = len(neighbors)

                # Check for mates
                for nid in neighbors[:5]:
                    if nid in self.arrays.id_to_index:
                        nidx = self.arrays.id_to_index[nid]
                        if self.arrays.energy[nidx] >= 50:
                            has_mate[idx] = True
                            break

            # FSM transitions
            new_states = self.fsm.compute_transitions(
                self.arrays, neighbor_counts, has_food, has_predator, has_mate
            )
            self.arrays.fsm_state[:n] = new_states

            # Compute movement
            target_x = np.zeros(n, dtype=np.float32)
            target_y = np.zeros(n, dtype=np.float32)
            has_target = np.zeros(n, dtype=np.bool_)

            dx, dy = self.fsm.compute_movement(
                self.arrays, target_x, target_y, has_target, self.rng
            )
            self.fsm.apply_movement(self.arrays, dx, dy)

            # Update energy
            food_consumed = np.zeros(n, dtype=np.float32)
            self.fsm.update_energy(self.arrays, food_consumed, self.config.BASE_ENERGY_DRAIN)
            self.fsm.update_age(self.arrays, self.config.MAX_AGE)

            # Check for stripe boundary crossings
            transfers = 0
            deaths = 0

            for idx in self.arrays.get_alive_indices():
                agent_id = self.arrays.ids[idx]
                y = self.arrays.y[idx]

                # Check if agent left stripe
                if y < self.stripe_start or y >= self.stripe_end:
                    # Transfer to another stripe
                    if agent_id in self.agents:
                        agent = self.agents.pop(agent_id)
                        # Update agent position from arrays
                        agent.x = float(self.arrays.x[idx])
                        agent.y = float(self.arrays.y[idx])
                        agent.energy = float(self.arrays.energy[idx])
                        agent.age = int(self.arrays.age[idx])
                        self.outgoing_agents.append(agent)
                        self.arrays.remove_agent(agent_id)
                        self.grid.remove(agent_id)
                        transfers += 1

                # Check deaths
                elif not self.arrays.alive[idx]:
                    if agent_id in self.agents:
                        self.agents.pop(agent_id)
                        self.arrays.remove_agent(agent_id)
                        self.grid.remove(agent_id)
                        deaths += 1

            # Update grid positions
            for idx in self.arrays.get_alive_indices():
                agent_id = self.arrays.ids[idx]
                self.grid.move(agent_id, self.arrays.x[idx], self.arrays.y[idx])

            # Sync back to agent objects
            self.arrays.sync_to_state_dict(self.agents)

            return {
                'births': 0,  # Reproduction handled by coordinator
                'deaths': deaths,
                'transfers': transfers,
                'population': len(self.agents),
            }

        def get_agent_ids(self) -> List[int]:
            """Get list of agent IDs in this stripe"""
            return list(self.agents.keys())

        def get_population(self) -> int:
            """Get current population in stripe"""
            return len(self.agents)

        def get_state(self) -> Dict[str, Any]:
            """Get full state for checkpointing"""
            return {
                'actor_id': self.actor_id,
                'stripe_start': self.stripe_start,
                'stripe_end': self.stripe_end,
                'agents': {aid: a.to_dict() for aid, a in self.agents.items()},
            }


class DistributedSimulation:
    """
    Coordinator for distributed simulation across multiple Ray actors.

    Manages:
    - Actor lifecycle
    - Agent transfers between stripes
    - Global state synchronization
    - Checkpointing
    """

    def __init__(self, config: SimConfig, num_actors: int = 8):
        if not RAY_AVAILABLE:
            raise ImportError("Ray is required for distributed simulation. Install with: pip install 'ray[default]'")

        self.config = config
        self.num_actors = num_actors

        # Initialize Ray if needed
        if not ray.is_initialized():
            ray.init()

        # Create stripe actors
        stripe_height = config.WORLD_SIZE / num_actors
        self.actors = []

        for i in range(num_actors):
            stripe_start = i * stripe_height
            stripe_end = (i + 1) * stripe_height

            actor = StripeActor.remote(
                actor_id=i,
                stripe_start=stripe_start,
                stripe_end=stripe_end,
                config=config
            )
            self.actors.append(actor)

        self.tick = 0

    def initialize(self, agents: List[AgentState]) -> None:
        """Distribute initial agents across stripes"""
        stripe_height = self.config.WORLD_SIZE / self.num_actors

        # Group agents by stripe
        stripe_agents = [[] for _ in range(self.num_actors)]

        for agent in agents:
            stripe_idx = int(agent.y / stripe_height)
            stripe_idx = min(stripe_idx, self.num_actors - 1)
            stripe_agents[stripe_idx].append(agent.to_dict())

        # Send to actors in parallel
        futures = []
        for i, actor in enumerate(self.actors):
            for agent_data in stripe_agents[i]:
                futures.append(actor.add_agent.remote(AgentState.from_dict(agent_data)))

        ray.get(futures)

    def step(self) -> Dict[str, Any]:
        """Execute one distributed tick"""
        # 1. All actors process their tick in parallel
        tick_futures = [actor.tick.remote(self.tick) for actor in self.actors]
        tick_results = ray.get(tick_futures)

        # 2. Collect outgoing agents from all actors
        outgoing_futures = [actor.get_outgoing_agents.remote() for actor in self.actors]
        all_outgoing = ray.get(outgoing_futures)

        # 3. Route agents to correct stripes
        stripe_height = self.config.WORLD_SIZE / self.num_actors
        incoming = [[] for _ in range(self.num_actors)]

        for agents in all_outgoing:
            for agent_data in agents:
                y = agent_data['y']
                stripe_idx = int(y / stripe_height) % self.num_actors
                incoming[stripe_idx].append(agent_data)

        # 4. Send incoming agents to actors
        receive_futures = [
            actor.receive_agents.remote(incoming[i])
            for i, actor in enumerate(self.actors)
        ]
        ray.get(receive_futures)

        # Aggregate stats
        total_stats = {
            'tick': self.tick,
            'births': sum(r['births'] for r in tick_results),
            'deaths': sum(r['deaths'] for r in tick_results),
            'transfers': sum(r['transfers'] for r in tick_results),
            'population': sum(r['population'] for r in tick_results),
        }

        self.tick += 1
        return total_stats

    def get_population(self) -> int:
        """Get total population across all actors"""
        pop_futures = [actor.get_population.remote() for actor in self.actors]
        populations = ray.get(pop_futures)
        return sum(populations)

    def shutdown(self) -> None:
        """Shutdown Ray actors"""
        for actor in self.actors:
            ray.kill(actor)


else:
    # Stub classes when Ray is not available
    class StripeActor:
        def __init__(self, *args, **kwargs):
            raise ImportError("Ray is required for distributed simulation")

    class DistributedSimulation:
        def __init__(self, *args, **kwargs):
            raise ImportError("Ray is required for distributed simulation. Install with: pip install 'ray[default]'")
