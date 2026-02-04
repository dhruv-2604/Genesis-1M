"""Tests for agent and world state"""

import pytest
import numpy as np
from src.core.state import AgentState, WorldState, AgentArrays, FSMState, GeneIndex


class TestAgentState:
    def test_create_agent(self):
        agent = AgentState(id=0, x=100.0, y=200.0)
        assert agent.id == 0
        assert agent.x == 100.0
        assert agent.y == 200.0
        assert agent.energy == 100.0
        assert agent.age == 0
        assert agent.is_alive

    def test_agent_genes(self):
        agent = AgentState(id=0, x=0, y=0)
        assert len(agent.genes) == GeneIndex.NUM_GENES
        assert 0.5 <= agent.speed <= 1.5
        assert 0.5 <= agent.metabolism <= 1.5

    def test_agent_serialization(self):
        agent = AgentState(
            id=42,
            x=100.0,
            y=200.0,
            energy=75.0,
            age=1000,
            parent_ids=(10, 20)
        )

        data = agent.to_dict()
        restored = AgentState.from_dict(data)

        assert restored.id == 42
        assert restored.x == 100.0
        assert restored.y == 200.0
        assert restored.energy == 75.0
        assert restored.age == 1000
        assert restored.parent_ids == (10, 20)

    def test_agent_death(self):
        agent = AgentState(id=0, x=0, y=0, energy=0)
        assert not agent.is_alive

    def test_reproduction_eligibility(self):
        # Young agent can't reproduce
        young = AgentState(id=0, x=0, y=0, energy=100, age=100)
        assert not young.can_reproduce

        # Low energy agent can't reproduce
        tired = AgentState(id=1, x=0, y=0, energy=20, age=10000)
        assert not tired.can_reproduce


class TestWorldState:
    def test_add_remove_agents(self):
        world = WorldState()

        agent1 = AgentState(id=0, x=100, y=100)
        agent2 = AgentState(id=1, x=200, y=200)

        world.add_agent(agent1)
        world.add_agent(agent2)

        assert world.population == 2
        assert world.get_agent(0) == agent1
        assert world.get_agent(1) == agent2

        world.remove_agent(0)
        assert world.population == 1
        assert world.get_agent(0) is None

    def test_world_serialization(self):
        world = WorldState(tick=100, world_size=5000.0)
        world.add_agent(AgentState(id=0, x=100, y=100))
        world.add_agent(AgentState(id=1, x=200, y=200))
        world.total_births = 10
        world.total_deaths = 5

        data = world.to_dict()
        restored = WorldState.from_dict(data)

        assert restored.tick == 100
        assert restored.world_size == 5000.0
        assert restored.population == 2
        assert restored.total_births == 10
        assert restored.total_deaths == 5


class TestAgentArrays:
    def test_add_agents(self):
        arrays = AgentArrays(max_agents=1000)

        agent1 = AgentState(id=0, x=100, y=200, energy=80)
        agent2 = AgentState(id=1, x=300, y=400, energy=60)

        arrays.add_agent(agent1)
        arrays.add_agent(agent2)

        assert arrays.count == 2
        assert 0 in arrays.id_to_index
        assert 1 in arrays.id_to_index

        idx0 = arrays.id_to_index[0]
        assert arrays.x[idx0] == 100
        assert arrays.y[idx0] == 200
        assert arrays.energy[idx0] == 80

    def test_remove_agent(self):
        arrays = AgentArrays(max_agents=1000)

        agent = AgentState(id=42, x=100, y=100)
        arrays.add_agent(agent)

        assert 42 in arrays.id_to_index
        assert arrays.alive[arrays.id_to_index[42]]

        arrays.remove_agent(42)

        assert 42 not in arrays.id_to_index
        assert len(arrays.free_slots) == 1

    def test_sync_to_state(self):
        arrays = AgentArrays(max_agents=1000)
        world = WorldState()

        agent = AgentState(id=0, x=100, y=100, energy=100)
        world.add_agent(agent)
        arrays.add_agent(agent)

        # Modify arrays
        idx = arrays.id_to_index[0]
        arrays.x[idx] = 500
        arrays.y[idx] = 600
        arrays.energy[idx] = 50

        # Sync back
        arrays.sync_to_state(world)

        assert world.agents[0].x == 500
        assert world.agents[0].y == 600
        assert world.agents[0].energy == 50

    def test_alive_mask(self):
        arrays = AgentArrays(max_agents=1000)

        for i in range(5):
            arrays.add_agent(AgentState(id=i, x=i*100, y=i*100))

        arrays.remove_agent(2)

        mask = arrays.get_alive_mask()
        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False
        assert mask[3] == True
        assert mask[4] == True
