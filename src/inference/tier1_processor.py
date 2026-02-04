"""Tier 1 Agent Processor - LLM-driven decision making"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.state import AgentState, AgentArrays, WorldState, AgentTier, FSMState, GeneIndex
from ..world.resources import ResourceType
from ..memory import MemoryManager, MemoryType
from .vllm_backend import VLLMBackend, MockVLLMBackend, InferenceResponse, create_backend
from .promotion import PromotionScorer, PromotionCandidate
from .prompts import build_prompt, build_system_prompt


@dataclass
class Tier1Action:
    """Parsed action from LLM response"""
    agent_id: int
    action_type: str  # move, eat, rest, approach, flee, trade, attack, mate, gather
    target_id: Optional[int] = None
    target_pos: Optional[Tuple[float, float]] = None
    speech: Optional[str] = None
    thought: Optional[str] = None


class Tier1Processor:
    """
    Processes Tier 1 (LLM-driven) agents.

    Workflow per tick:
    1. Collect responses from PREVIOUS tick's batch
    2. Apply actions from those responses
    3. Identify new promotion candidates
    4. Submit new batch for NEXT tick

    This 1-tick delay allows batching without blocking.
    """

    def __init__(
        self,
        backend: Optional[VLLMBackend] = None,
        scorer: Optional[PromotionScorer] = None,
        memory_manager: Optional[MemoryManager] = None,
        use_mock: bool = False,
        promotion_budget: int = 500,
    ):
        self.backend = backend or create_backend(use_mock=use_mock)
        self.scorer = scorer or PromotionScorer(global_budget=promotion_budget)
        self.memory = memory_manager or MemoryManager(use_mock=use_mock)

        # Track active Tier 1 agents
        self.tier1_agents: Dict[int, Dict[str, Any]] = {}

        # Pending actions from last tick
        self.pending_actions: Dict[int, Tier1Action] = {}

        # Stats
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_actions_applied = 0
        self.total_memories_created = 0

    def process_tick(
        self,
        arrays: AgentArrays,
        world_state: WorldState,
        neighbor_data: Dict[int, List[int]],
        resource_data: Dict[int, Dict[str, float]],
        terrain_data: Dict[int, str],
    ) -> Tuple[List[Tier1Action], List[int], List[int]]:
        """
        Process one tick for Tier 1 agents.

        Returns:
            - actions: List of actions to apply this tick
            - promoted: List of agent IDs newly promoted to Tier 1
            - demoted: List of agent IDs demoted back to Tier 3
        """
        current_tick = world_state.tick

        # 1. Collect responses from previous tick's batch
        responses = self.backend.process_batch(current_tick)

        # 2. Parse responses into actions
        actions = []
        for response in responses:
            action = self._parse_response(response)
            if action:
                actions.append(action)
                self.pending_actions[action.agent_id] = action

        # 3. Calculate promotion scores
        resource_seekers = self._get_resource_seekers(arrays)
        candidates = self.scorer.calculate_scores(
            arrays, neighbor_data, resource_seekers, current_tick
        )

        # 4. Select promotions within budget
        promoted_ids = self.scorer.select_promotions(candidates, current_tick)

        # 5. Submit new inference requests for promoted + existing Tier 1
        all_tier1_ids = set(promoted_ids) | set(self.tier1_agents.keys())

        for agent_id in all_tier1_ids:
            if agent_id not in arrays.id_to_index:
                continue

            idx = arrays.id_to_index[agent_id]
            if not arrays.alive[idx]:
                continue

            # Build context for this agent
            agent_data = self._get_agent_data(arrays, idx, agent_id)
            neighbors = neighbor_data.get(agent_id, [])
            neighbor_info = self._get_neighbor_info(arrays, neighbors)
            resources = resource_data.get(agent_id, {})
            terrain = terrain_data.get(agent_id, "PLAINS")

            # Get recent memories for prompt context
            memories = self._get_memories(agent_id, neighbors, current_tick)

            # Build prompts
            system_prompt = build_system_prompt(agent_data)
            user_prompt = build_prompt(
                agent_state=agent_data,
                nearby_agents=neighbor_info,
                nearby_resources=resources,
                terrain_type=terrain,
                memories=memories,
            )

            # Calculate priority
            priority = 1.0
            for c in candidates:
                if c.agent_id == agent_id:
                    priority = c.score
                    break

            # Submit request
            self.backend.submit_request(
                agent_id=agent_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tick=current_tick,
                priority=priority,
            )

        # 6. Update tier tracking
        for agent_id in promoted_ids:
            if agent_id not in self.tier1_agents:
                self.tier1_agents[agent_id] = {"promoted_tick": current_tick}
                self.total_promotions += 1

                # Update array
                if agent_id in arrays.id_to_index:
                    arrays.tier[arrays.id_to_index[agent_id]] = AgentTier.TIER1

        # 7. Check for demotions (agents inactive for too long)
        demoted_ids = []
        for agent_id in list(self.tier1_agents.keys()):
            if agent_id not in arrays.id_to_index:
                # Agent died
                del self.tier1_agents[agent_id]
                demoted_ids.append(agent_id)
                continue

            info = self.tier1_agents[agent_id]
            ticks_active = current_tick - info["promoted_tick"]

            # Demote after 10 ticks of no interesting activity
            if ticks_active > 10 and agent_id not in promoted_ids:
                idx = arrays.id_to_index[agent_id]
                arrays.tier[idx] = AgentTier.TIER3
                del self.tier1_agents[agent_id]
                demoted_ids.append(agent_id)
                self.total_demotions += 1

        self.total_actions_applied += len(actions)

        return actions, promoted_ids, demoted_ids

    def apply_action(
        self,
        action: Tier1Action,
        arrays: AgentArrays,
        world_state: WorldState,
    ) -> bool:
        """Apply a Tier 1 action to the simulation"""
        if action.agent_id not in arrays.id_to_index:
            return False

        idx = arrays.id_to_index[action.agent_id]
        if not arrays.alive[idx]:
            return False

        action_type = action.action_type.lower()

        if action_type in ["move", "wander"]:
            # Random movement
            arrays.fsm_state[idx] = FSMState.WANDER

        elif action_type == "eat":
            arrays.fsm_state[idx] = FSMState.SEEK_FOOD

        elif action_type == "rest":
            arrays.fsm_state[idx] = FSMState.REST

        elif action_type in ["approach", "follow"]:
            if action.target_id and action.target_id in arrays.id_to_index:
                tidx = arrays.id_to_index[action.target_id]
                # Record social interaction
                self.memory.record_interaction(
                    agent_id=action.agent_id,
                    other_agent_id=action.target_id,
                    tick=world_state.tick,
                    action=f"approached",
                    outcome="Moving closer",
                    valence=0.1,
                    location=(float(arrays.x[idx]), float(arrays.y[idx])),
                )
                self.total_memories_created += 1
            arrays.fsm_state[idx] = FSMState.WANDER

        elif action_type == "flee":
            arrays.fsm_state[idx] = FSMState.FLEE

        elif action_type == "mate":
            arrays.fsm_state[idx] = FSMState.SEEK_MATE

        elif action_type in ["gather", "forage"]:
            arrays.fsm_state[idx] = FSMState.SEEK_FOOD

        elif action_type == "attack":
            # Combat system (simplified for now)
            if action.target_id and action.target_id in arrays.id_to_index:
                tidx = arrays.id_to_index[action.target_id]

                # Attacker strength vs defender
                attacker_str = arrays.genes[idx, GeneIndex.STRENGTH]
                defender_str = arrays.genes[tidx, GeneIndex.STRENGTH]

                # Simple combat: both lose energy, weaker loses more
                arrays.energy[idx] -= 10
                arrays.energy[tidx] -= 10 + (attacker_str - defender_str) * 20

                # Record conflict memory for both parties
                won = attacker_str >= defender_str
                self.memory.record_conflict(
                    agent_id=action.agent_id,
                    other_agent_id=action.target_id,
                    tick=world_state.tick,
                    won=won,
                    location=(float(arrays.x[idx]), float(arrays.y[idx])),
                )
                self.memory.record_conflict(
                    agent_id=action.target_id,
                    other_agent_id=action.agent_id,
                    tick=world_state.tick,
                    won=not won,
                    location=(float(arrays.x[tidx]), float(arrays.y[tidx])),
                )
                self.total_memories_created += 2

        elif action_type in ["offer_trade", "trade"]:
            # Trading system placeholder - record interaction
            if action.target_id and action.target_id in arrays.id_to_index:
                tidx = arrays.id_to_index[action.target_id]
                self.memory.record_interaction(
                    agent_id=action.agent_id,
                    other_agent_id=action.target_id,
                    tick=world_state.tick,
                    action="offered to trade",
                    outcome="Trade initiated",
                    valence=0.2,
                    location=(float(arrays.x[idx]), float(arrays.y[idx])),
                )
                self.total_memories_created += 1

        # Log speech as an observation memory
        if action.speech and action.target_id:
            self.memory.record_interaction(
                agent_id=action.agent_id,
                other_agent_id=action.target_id,
                tick=world_state.tick,
                action=f"said: '{action.speech[:50]}'",
                outcome="Spoke to another",
                valence=0.0,
                location=(float(arrays.x[idx]), float(arrays.y[idx])),
            )
            self.total_memories_created += 1

        return True

    def _parse_response(self, response: InferenceResponse) -> Optional[Tier1Action]:
        """Parse LLM response into action"""
        if not response.parsed_action:
            return None

        parsed = response.parsed_action

        return Tier1Action(
            agent_id=response.agent_id,
            action_type=parsed.get("action", "wander"),
            target_id=parsed.get("target_id"),
            speech=parsed.get("speech"),
            thought=parsed.get("thought"),
        )

    def _get_agent_data(self, arrays: AgentArrays, idx: int, agent_id: int) -> Dict:
        """Extract agent data for prompt, including personality genes"""
        return {
            "id": agent_id,
            "name": f"Human-{agent_id}",
            "energy": float(arrays.energy[idx]),
            "age": int(arrays.age[idx]),
            "fsm_state": int(arrays.fsm_state[idx]),
            "genes": arrays.genes[idx].tolist(),  # For personality injection
            "inventory": {},  # TODO: integrate inventory
        }

    def _get_neighbor_info(self, arrays: AgentArrays, neighbor_ids: List[int]) -> List[Dict]:
        """Get info about neighboring agents"""
        neighbors = []
        for nid in neighbor_ids[:5]:  # Limit to 5
            if nid in arrays.id_to_index:
                idx = arrays.id_to_index[nid]
                neighbors.append({
                    "id": nid,
                    "name": f"Human-{nid}",
                    "energy": float(arrays.energy[idx]),
                    "fsm_state": int(arrays.fsm_state[idx]),
                })
        return neighbors

    def _get_resource_seekers(self, arrays: AgentArrays) -> set:
        """Get set of agents currently seeking resources"""
        seekers = set()
        for idx in arrays.get_alive_indices():
            if arrays.fsm_state[idx] == FSMState.SEEK_FOOD:
                seekers.add(int(arrays.ids[idx]))
        return seekers

    def _get_memories(self, agent_id: int, nearby_ids: List[int], current_tick: int) -> List[str]:
        """Get recent memories for agent"""
        return self.memory.recall_for_prompt(
            agent_id=agent_id,
            current_tick=current_tick,
            nearby_agent_ids=nearby_ids,
            limit=5,
        )

    def flush_memories(self) -> int:
        """Flush pending memories to storage. Call at end of tick."""
        return self.memory.flush_pending()

    def handle_agent_death(
        self,
        agent_id: int,
        child_ids: List[int],
        current_tick: int,
    ) -> int:
        """Handle memory inheritance when a Tier 1 agent dies."""
        # Remove from tier1 tracking
        if agent_id in self.tier1_agents:
            del self.tier1_agents[agent_id]

        # Transfer memories to children
        return self.memory.handle_agent_death(
            agent_id=agent_id,
            child_ids=child_ids,
            current_tick=current_tick,
        )

    def get_stats(self) -> Dict:
        return {
            "active_tier1": len(self.tier1_agents),
            "total_promotions": self.total_promotions,
            "total_demotions": self.total_demotions,
            "total_actions_applied": self.total_actions_applied,
            "total_memories_created": self.total_memories_created,
            "backend_stats": self.backend.get_stats(),
            "scorer_stats": self.scorer.get_stats(),
            "memory_stats": self.memory.get_stats(),
        }
