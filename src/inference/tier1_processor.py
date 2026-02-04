"""Tier 1 Agent Processor - LLM-driven decision making"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from ..core.state import AgentState, AgentArrays, WorldState, AgentTier, FSMState, GeneIndex
from ..world.resources import ResourceType
from .vllm_backend import VLLMBackend, MockVLLMBackend, InferenceResponse, create_backend
from .promotion import PromotionScorer, PromotionCandidate
from .prompts import build_prompt, build_system_prompt, PRIMITIVE_HUMAN_SYSTEM


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
        use_mock: bool = False,
        promotion_budget: int = 500,
    ):
        self.backend = backend or create_backend(use_mock=use_mock)
        self.scorer = scorer or PromotionScorer(global_budget=promotion_budget)

        # Track active Tier 1 agents
        self.tier1_agents: Dict[int, Dict[str, Any]] = {}

        # Pending actions from last tick
        self.pending_actions: Dict[int, Tier1Action] = {}

        # Stats
        self.total_promotions = 0
        self.total_demotions = 0
        self.total_actions_applied = 0

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

            # Get recent memories (placeholder for Phase 3)
            memories = self._get_memories(agent_id)

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
                # Set target position
                # (actual movement handled by FSM)
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

        elif action_type in ["offer_trade", "trade"]:
            # Trading system placeholder
            pass

        # Log speech if present
        if action.speech:
            # TODO: Log to event system
            pass

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
        """Extract agent data for prompt"""
        return {
            "id": agent_id,
            "name": f"Human-{agent_id}",
            "energy": float(arrays.energy[idx]),
            "age": int(arrays.age[idx]),
            "fsm_state": int(arrays.fsm_state[idx]),
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

    def _get_memories(self, agent_id: int) -> List[str]:
        """Get recent memories for agent (Phase 3)"""
        # Placeholder - will integrate with LanceDB
        return []

    def get_stats(self) -> Dict:
        return {
            "active_tier1": len(self.tier1_agents),
            "total_promotions": self.total_promotions,
            "total_demotions": self.total_demotions,
            "total_actions_applied": self.total_actions_applied,
            "backend_stats": self.backend.get_stats(),
            "scorer_stats": self.scorer.get_stats(),
        }
