"""Tests for Tier 3 FSM logic"""

import pytest
import numpy as np
from src.core.state import AgentState, AgentArrays, FSMState, GeneIndex
from src.agents.tier3 import Tier3FSM, crossover, mutate, create_child


class TestTier3FSM:
    @pytest.fixture
    def fsm(self):
        return Tier3FSM(world_size=1000, cell_size=50)

    @pytest.fixture
    def arrays(self):
        arrays = AgentArrays(max_agents=100)

        # Create some test agents
        for i in range(10):
            agent = AgentState(
                id=i,
                x=i * 100,
                y=i * 100,
                energy=50 + i * 5,
                age=i * 1000
            )
            arrays.add_agent(agent)

        return arrays

    def test_transition_to_seek_food(self, fsm, arrays):
        # Set low energy for agent 0
        idx = arrays.id_to_index[0]
        arrays.energy[idx] = 20  # Below hunger threshold

        n = arrays.count
        neighbor_counts = np.zeros(n, dtype=np.int32)
        has_food = np.ones(n, dtype=np.bool_)
        has_predator = np.zeros(n, dtype=np.bool_)
        has_mate = np.zeros(n, dtype=np.bool_)

        new_states = fsm.compute_transitions(
            arrays, neighbor_counts, has_food, has_predator, has_mate
        )

        assert new_states[idx] == FSMState.SEEK_FOOD

    def test_transition_to_flee(self, fsm, arrays):
        idx = arrays.id_to_index[0]

        n = arrays.count
        neighbor_counts = np.zeros(n, dtype=np.int32)
        has_food = np.zeros(n, dtype=np.bool_)
        has_predator = np.zeros(n, dtype=np.bool_)
        has_predator[idx] = True  # Predator nearby
        has_mate = np.zeros(n, dtype=np.bool_)

        new_states = fsm.compute_transitions(
            arrays, neighbor_counts, has_food, has_predator, has_mate
        )

        assert new_states[idx] == FSMState.FLEE

    def test_transition_to_seek_mate(self, fsm, arrays):
        idx = arrays.id_to_index[5]  # Agent with enough age

        # Set up for mating
        arrays.energy[idx] = 80
        arrays.age[idx] = 10000  # Above maturity
        arrays.reproductive_cooldown[idx] = 0

        n = arrays.count
        neighbor_counts = np.ones(n, dtype=np.int32)
        has_food = np.zeros(n, dtype=np.bool_)
        has_predator = np.zeros(n, dtype=np.bool_)
        has_mate = np.zeros(n, dtype=np.bool_)
        has_mate[idx] = True

        new_states = fsm.compute_transitions(
            arrays, neighbor_counts, has_food, has_predator, has_mate
        )

        assert new_states[idx] == FSMState.SEEK_MATE

    def test_transition_to_rest(self, fsm, arrays):
        idx = arrays.id_to_index[0]
        arrays.energy[idx] = 95  # Above rest threshold

        n = arrays.count
        neighbor_counts = np.zeros(n, dtype=np.int32)
        has_food = np.zeros(n, dtype=np.bool_)
        has_predator = np.zeros(n, dtype=np.bool_)
        has_mate = np.zeros(n, dtype=np.bool_)

        new_states = fsm.compute_transitions(
            arrays, neighbor_counts, has_food, has_predator, has_mate
        )

        assert new_states[idx] == FSMState.REST

    def test_movement_toward_target(self, fsm, arrays):
        n = arrays.count
        rng = np.random.default_rng(42)

        target_x = np.zeros(n, dtype=np.float32)
        target_y = np.zeros(n, dtype=np.float32)
        has_target = np.zeros(n, dtype=np.bool_)

        # Set target for agent 0
        idx = arrays.id_to_index[0]
        arrays.x[idx] = 100
        arrays.y[idx] = 100
        target_x[idx] = 200  # Target to the right
        target_y[idx] = 100
        has_target[idx] = True
        arrays.fsm_state[idx] = FSMState.SEEK_FOOD

        dx, dy = fsm.compute_movement(arrays, target_x, target_y, has_target, rng)

        # Should move right (positive dx)
        assert dx[idx] > 0
        assert abs(dy[idx]) < 0.1  # Minimal y movement

    def test_apply_movement(self, fsm, arrays):
        idx = arrays.id_to_index[0]
        arrays.x[idx] = 500
        arrays.y[idx] = 500

        n = arrays.count
        dx = np.zeros(n, dtype=np.float32)
        dy = np.zeros(n, dtype=np.float32)
        dx[idx] = 10
        dy[idx] = -5

        fsm.apply_movement(arrays, dx, dy)

        assert arrays.x[idx] == 510
        assert arrays.y[idx] == 495

    def test_movement_wraps_around(self, fsm, arrays):
        idx = arrays.id_to_index[0]
        arrays.x[idx] = 995
        arrays.y[idx] = 995

        n = arrays.count
        dx = np.zeros(n, dtype=np.float32)
        dy = np.zeros(n, dtype=np.float32)
        dx[idx] = 10
        dy[idx] = 10

        fsm.apply_movement(arrays, dx, dy)

        # Should wrap around
        assert arrays.x[idx] == 5
        assert arrays.y[idx] == 5

    def test_energy_drain(self, fsm, arrays):
        initial_energy = arrays.energy.copy()

        n = arrays.count
        food_consumed = np.zeros(n, dtype=np.float32)

        fsm.update_energy(arrays, food_consumed, base_drain=1.0)

        # All agents should have less energy
        for idx in arrays.get_alive_indices():
            assert arrays.energy[idx] < initial_energy[idx]

    def test_energy_from_food(self, fsm, arrays):
        idx = arrays.id_to_index[0]
        initial_energy = arrays.energy[idx]

        n = arrays.count
        food_consumed = np.zeros(n, dtype=np.float32)
        food_consumed[idx] = 20.0  # Ate food

        fsm.update_energy(arrays, food_consumed, base_drain=1.0)

        # Should have more energy (food > drain)
        assert arrays.energy[idx] > initial_energy

    def test_age_increment(self, fsm, arrays):
        initial_ages = arrays.age.copy()

        fsm.update_age(arrays, max_age=30000)

        for idx in arrays.get_alive_indices():
            assert arrays.age[idx] == initial_ages[idx] + 1

    def test_death_by_age(self, fsm, arrays):
        # Set agent to max age
        idx = arrays.id_to_index[0]
        arrays.age[idx] = 30000

        fsm.update_age(arrays, max_age=30000)

        assert not arrays.alive[idx]


class TestGenetics:
    def test_crossover(self):
        rng = np.random.default_rng(42)

        parent1 = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
        parent2 = np.array([1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8])

        child = crossover(parent1, parent2, rng)

        assert len(child) == len(parent1)
        # Child should have mix of both parents' genes
        # Can't test exactly due to random crossover point

    def test_mutate(self):
        rng = np.random.default_rng(42)
        genes = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        mutated = mutate(genes, rng, mutation_rate=1.0, mutation_strength=0.1)

        # With 100% mutation rate, some genes should change
        assert not np.allclose(genes, mutated)

    def test_mutate_clamping(self):
        rng = np.random.default_rng(42)
        genes = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

        mutated = mutate(genes, rng, mutation_rate=1.0, mutation_strength=0.5)

        # Should be clamped to max 2.0
        assert np.all(mutated <= 2.0)
        assert np.all(mutated >= 0.0)

    def test_create_child(self):
        rng = np.random.default_rng(42)

        parent1_genes = np.array([1.0] * 8)
        parent2_genes = np.array([1.5] * 8)

        child_data = create_child(
            parent1_genes, parent2_genes,
            parent1_id=10, parent2_id=20,
            x=100, y=200,
            new_id=30,
            rng=rng
        )

        assert child_data['id'] == 30
        assert child_data['x'] == 100
        assert child_data['y'] == 200
        assert child_data['parent_ids'] == (10, 20)
        assert child_data['energy'] == 50.0
        assert child_data['age'] == 0
        assert len(child_data['genes']) == 8
