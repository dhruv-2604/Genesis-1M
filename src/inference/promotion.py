"""Promotion Scoring System for Tier Transitions"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np

from ..core.state import AgentState, AgentArrays, AgentTier, GeneIndex


@dataclass
class PromotionCandidate:
    """Agent being considered for promotion"""
    agent_id: int
    score: float
    reasons: List[str]


class PromotionScorer:
    """
    Calculates promotion priority scores for agents.

    Higher scores = more interesting situations = more likely to get LLM attention.
    Uses a global budget to limit LLM calls per tick.
    """

    # Score weights
    NEIGHBOR_WEIGHT = 0.1       # Per neighbor
    RESOURCE_CONFLICT_WEIGHT = 2.0
    NOVEL_ENCOUNTER_WEIGHT = 1.5
    LOW_ENERGY_WEIGHT = 1.0
    INTELLIGENCE_WEIGHT = 0.5   # Gene bonus
    HIGH_DENSITY_WEIGHT = 0.3
    MATING_OPPORTUNITY_WEIGHT = 1.0

    # Thresholds
    LOW_ENERGY_THRESHOLD = 20.0
    HIGH_DENSITY_THRESHOLD = 5

    def __init__(
        self,
        global_budget: int = 500,
        min_score_threshold: float = 1.0,
        cooldown_ticks: int = 10,
    ):
        self.global_budget = global_budget
        self.min_score_threshold = min_score_threshold
        self.cooldown_ticks = cooldown_ticks

        # Track recently promoted agents to avoid spam
        self.promotion_cooldowns: Dict[int, int] = {}

        # Track known encounters to detect novel ones
        self.known_encounters: Dict[int, Set[int]] = {}

    def calculate_scores(
        self,
        arrays: AgentArrays,
        neighbor_data: Dict[int, List[int]],  # agent_id -> neighbor_ids
        resource_seekers: Set[int],  # agents seeking same resource
        current_tick: int,
    ) -> List[PromotionCandidate]:
        """
        Calculate promotion scores for all eligible agents.
        Returns sorted list of candidates (highest score first).
        """
        candidates = []

        # Update cooldowns
        expired = [aid for aid, tick in self.promotion_cooldowns.items()
                   if current_tick - tick >= self.cooldown_ticks]
        for aid in expired:
            del self.promotion_cooldowns[aid]

        alive_indices = arrays.get_alive_indices()

        for idx in alive_indices:
            agent_id = int(arrays.ids[idx])

            # Skip if on cooldown
            if agent_id in self.promotion_cooldowns:
                continue

            # Skip if already Tier 1
            if arrays.tier[idx] == AgentTier.TIER1:
                continue

            score = 0.0
            reasons = []

            neighbors = neighbor_data.get(agent_id, [])

            # 1. Social density (more neighbors = more interesting)
            if len(neighbors) > 0:
                density_score = len(neighbors) * self.NEIGHBOR_WEIGHT
                score += density_score
                if len(neighbors) >= self.HIGH_DENSITY_THRESHOLD:
                    score += self.HIGH_DENSITY_WEIGHT
                    reasons.append(f"high_density({len(neighbors)})")

            # 2. Resource conflict (multiple agents, same goal)
            if agent_id in resource_seekers:
                conflict_neighbors = [n for n in neighbors if n in resource_seekers]
                if conflict_neighbors:
                    score += self.RESOURCE_CONFLICT_WEIGHT
                    reasons.append("resource_conflict")

            # 3. Novel encounter (never met this agent before)
            known = self.known_encounters.get(agent_id, set())
            novel = [n for n in neighbors if n not in known]
            if novel:
                score += self.NOVEL_ENCOUNTER_WEIGHT * min(len(novel), 3)
                reasons.append(f"novel_encounter({len(novel)})")

            # 4. Desperation (low energy = interesting decisions)
            energy = arrays.energy[idx]
            if energy < self.LOW_ENERGY_THRESHOLD:
                desperation = (self.LOW_ENERGY_THRESHOLD - energy) / self.LOW_ENERGY_THRESHOLD
                score += self.LOW_ENERGY_WEIGHT * desperation
                reasons.append(f"low_energy({energy:.0f})")

            # 5. Intelligence gene bonus
            intelligence = arrays.genes[idx, GeneIndex.INTELLIGENCE]
            score += intelligence * self.INTELLIGENCE_WEIGHT

            # 6. Mating opportunity
            if arrays.reproductive_cooldown[idx] == 0 and energy >= 50:
                for nid in neighbors:
                    if nid in arrays.id_to_index:
                        nidx = arrays.id_to_index[nid]
                        if (arrays.energy[nidx] >= 50 and
                            arrays.reproductive_cooldown[nidx] == 0):
                            score += self.MATING_OPPORTUNITY_WEIGHT
                            reasons.append("mating_opportunity")
                            break

            # Only consider if above threshold
            if score >= self.min_score_threshold:
                candidates.append(PromotionCandidate(
                    agent_id=agent_id,
                    score=score,
                    reasons=reasons,
                ))

        # Sort by score (descending)
        candidates.sort(key=lambda c: c.score, reverse=True)

        return candidates

    def select_promotions(
        self,
        candidates: List[PromotionCandidate],
        current_tick: int,
    ) -> List[int]:
        """
        Select agents to promote based on budget.
        Returns list of agent IDs to promote.
        """
        promoted = []

        for candidate in candidates[:self.global_budget]:
            promoted.append(candidate.agent_id)
            self.promotion_cooldowns[candidate.agent_id] = current_tick

        return promoted

    def record_encounter(self, agent_id: int, other_id: int) -> None:
        """Record that two agents have met"""
        if agent_id not in self.known_encounters:
            self.known_encounters[agent_id] = set()
        self.known_encounters[agent_id].add(other_id)

        if other_id not in self.known_encounters:
            self.known_encounters[other_id] = set()
        self.known_encounters[other_id].add(agent_id)

    def demote_agent(self, agent_id: int) -> None:
        """Mark agent for demotion back to Tier 3"""
        # Cooldown prevents immediate re-promotion
        pass

    def get_stats(self) -> Dict:
        return {
            "agents_on_cooldown": len(self.promotion_cooldowns),
            "known_encounter_pairs": sum(len(v) for v in self.known_encounters.values()) // 2,
            "budget": self.global_budget,
        }
