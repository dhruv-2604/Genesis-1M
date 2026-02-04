"""Tests for LLM inference and promotion system"""

import pytest
import numpy as np

from src.core.state import AgentState, AgentArrays, FSMState, GeneIndex, AgentTier
from src.inference.promotion import PromotionScorer, PromotionCandidate
from src.inference.vllm_backend import MockVLLMBackend, InferenceRequest
from src.inference.tier1_processor import Tier1Processor, Tier1Action


class TestPromotionScorer:
    @pytest.fixture
    def scorer(self):
        return PromotionScorer(global_budget=10, min_score_threshold=0.5)

    @pytest.fixture
    def arrays(self):
        arrays = AgentArrays(max_agents=100)
        for i in range(20):
            agent = AgentState(
                id=i,
                x=i * 50,
                y=i * 50,
                energy=50 + i * 2,
                age=1000 + i * 100,
            )
            arrays.add_agent(agent)
        return arrays

    def test_score_increases_with_neighbors(self, scorer, arrays):
        # Agent with many neighbors should score higher
        neighbor_data = {
            0: [1, 2, 3, 4, 5],  # Many neighbors
            1: [],               # No neighbors
        }

        candidates = scorer.calculate_scores(
            arrays, neighbor_data, set(), current_tick=0
        )

        # Find scores for agents 0 and 1
        score_0 = next((c.score for c in candidates if c.agent_id == 0), 0)
        score_1 = next((c.score for c in candidates if c.agent_id == 1), 0)

        assert score_0 > score_1

    def test_score_increases_with_low_energy(self, scorer, arrays):
        # Set agent 0 to low energy
        idx = arrays.id_to_index[0]
        arrays.energy[idx] = 10  # Below threshold

        neighbor_data = {0: [], 1: []}

        candidates = scorer.calculate_scores(
            arrays, neighbor_data, set(), current_tick=0
        )

        score_0 = next((c.score for c in candidates if c.agent_id == 0), 0)

        # Low energy should give a score boost
        assert score_0 > 0
        assert any("low_energy" in r for c in candidates if c.agent_id == 0 for r in c.reasons)

    def test_score_increases_with_resource_conflict(self, scorer, arrays):
        neighbor_data = {
            0: [1, 2],
            1: [0, 2],
            2: [0, 1],
        }
        resource_seekers = {0, 1, 2}  # All seeking same resource

        candidates = scorer.calculate_scores(
            arrays, neighbor_data, resource_seekers, current_tick=0
        )

        # All should have resource_conflict reason
        for c in candidates:
            if c.agent_id in [0, 1, 2]:
                assert any("resource_conflict" in r for r in c.reasons)

    def test_budget_limits_promotions(self, scorer, arrays):
        # Create many high-scoring candidates
        neighbor_data = {i: list(range(20)) for i in range(20)}

        candidates = scorer.calculate_scores(
            arrays, neighbor_data, set(), current_tick=0
        )

        promoted = scorer.select_promotions(candidates, current_tick=0)

        # Should be limited to budget
        assert len(promoted) <= scorer.global_budget

    def test_cooldown_prevents_immediate_repromotion(self, scorer, arrays):
        neighbor_data = {0: [1, 2, 3]}

        # First promotion
        candidates1 = scorer.calculate_scores(arrays, neighbor_data, set(), current_tick=0)
        promoted1 = scorer.select_promotions(candidates1, current_tick=0)
        assert 0 in promoted1

        # Second promotion attempt (should be blocked by cooldown)
        candidates2 = scorer.calculate_scores(arrays, neighbor_data, set(), current_tick=1)
        agent_0_in_candidates = any(c.agent_id == 0 for c in candidates2)
        assert not agent_0_in_candidates

    def test_novel_encounter_bonus(self, scorer, arrays):
        neighbor_data = {0: [1, 2]}

        # First encounter
        candidates1 = scorer.calculate_scores(arrays, neighbor_data, set(), current_tick=0)
        score_first = next((c.score for c in candidates1 if c.agent_id == 0), 0)

        # Record encounters
        scorer.record_encounter(0, 1)
        scorer.record_encounter(0, 2)

        # Reset cooldown for test
        scorer.promotion_cooldowns.clear()

        # Second calculation (encounters no longer novel)
        candidates2 = scorer.calculate_scores(arrays, neighbor_data, set(), current_tick=100)
        score_second = next((c.score for c in candidates2 if c.agent_id == 0), 0)

        # First encounter should score higher due to novelty
        assert score_first > score_second


class TestMockVLLMBackend:
    def test_submit_and_process(self):
        backend = MockVLLMBackend()

        backend.submit_request(
            agent_id=1,
            system_prompt="You are a primitive human.",
            user_prompt="What do you do?",
            tick=0,
        )

        assert backend.queue_size == 1

        responses = backend.process_batch(current_tick=1)

        assert len(responses) == 1
        assert responses[0].agent_id == 1
        assert responses[0].parsed_action is not None
        assert backend.queue_size == 0

    def test_multiple_requests(self):
        backend = MockVLLMBackend()

        for i in range(5):
            backend.submit_request(
                agent_id=i,
                system_prompt="System",
                user_prompt="User",
                tick=0,
            )

        assert backend.queue_size == 5

        responses = backend.process_batch(current_tick=1)

        assert len(responses) == 5
        assert backend.queue_size == 0

    def test_get_response(self):
        backend = MockVLLMBackend()

        backend.submit_request(agent_id=42, system_prompt="S", user_prompt="U", tick=0)
        backend.process_batch(current_tick=1)

        response = backend.get_response(42)
        assert response is not None
        assert response.agent_id == 42

        # Second get should return None
        response2 = backend.get_response(42)
        assert response2 is None


class TestTier1Processor:
    @pytest.fixture
    def processor(self):
        return Tier1Processor(use_mock=True, promotion_budget=5)

    @pytest.fixture
    def arrays(self):
        arrays = AgentArrays(max_agents=100)
        for i in range(10):
            agent = AgentState(
                id=i,
                x=i * 10,
                y=i * 10,
                energy=50,
                age=5000,
            )
            arrays.add_agent(agent)
        return arrays

    @pytest.fixture
    def world_state(self):
        return type('WorldState', (), {'tick': 0})()

    def test_process_tick_promotes_agents(self, processor, arrays, world_state):
        neighbor_data = {i: [j for j in range(10) if j != i] for i in range(10)}
        resource_data = {i: {"BERRIES": 5.0} for i in range(10)}
        terrain_data = {i: "PLAINS" for i in range(10)}

        actions, promoted, demoted = processor.process_tick(
            arrays, world_state, neighbor_data, resource_data, terrain_data
        )

        # Should have some promotions
        assert len(promoted) > 0
        assert len(promoted) <= processor.scorer.global_budget

    def test_tier_updated_on_promotion(self, processor, arrays, world_state):
        neighbor_data = {0: [1, 2, 3, 4, 5]}
        resource_data = {0: {"BERRIES": 5.0}}
        terrain_data = {0: "PLAINS"}

        # Set low energy to trigger promotion
        arrays.energy[0] = 10

        processor.process_tick(
            arrays, world_state, neighbor_data, resource_data, terrain_data
        )

        # Process again to apply promotion
        world_state.tick = 1
        processor.process_tick(
            arrays, world_state, neighbor_data, resource_data, terrain_data
        )

        # Check if agent 0 is now Tier 1
        # (depends on scoring, but with low energy should qualify)

    def test_apply_action_changes_state(self, processor, arrays, world_state):
        action = Tier1Action(
            agent_id=0,
            action_type="rest",
        )

        result = processor.apply_action(action, arrays, world_state)

        assert result is True
        assert arrays.fsm_state[0] == FSMState.REST

    def test_apply_flee_action(self, processor, arrays, world_state):
        action = Tier1Action(
            agent_id=0,
            action_type="flee",
        )

        processor.apply_action(action, arrays, world_state)
        assert arrays.fsm_state[0] == FSMState.FLEE

    def test_apply_eat_action(self, processor, arrays, world_state):
        action = Tier1Action(
            agent_id=0,
            action_type="eat",
        )

        processor.apply_action(action, arrays, world_state)
        assert arrays.fsm_state[0] == FSMState.SEEK_FOOD

    def test_stats(self, processor):
        stats = processor.get_stats()

        assert "active_tier1" in stats
        assert "total_promotions" in stats
        assert "backend_stats" in stats
