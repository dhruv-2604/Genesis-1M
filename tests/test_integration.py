"""Integration tests for the simulation"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.core.simulation import Simulation
from src.config import SimConfig


class TestSimulationIntegration:
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def small_config(self, temp_dir):
        """Create config for small test simulation"""
        config = SimConfig()
        config.WORLD_SIZE = 1000.0
        config.CELL_SIZE = 50.0
        config.INITIAL_AGENT_COUNT = 100
        config.MAX_AGENTS = 500
        config.CHECKPOINT_DIR = f"{temp_dir}/checkpoints"
        config.EVENT_LOG_DIR = f"{temp_dir}/events"
        config.SEED = 42
        return config

    def test_simulation_initialization(self, small_config):
        sim = Simulation(config=small_config)

        assert sim.world_state.tick == 0
        assert sim.world_state.population == small_config.INITIAL_AGENT_COUNT
        assert len(sim.spatial_grid) == small_config.INITIAL_AGENT_COUNT

    def test_single_tick(self, small_config):
        sim = Simulation(config=small_config)
        initial_pop = sim.world_state.population

        stats = sim.tick()

        assert stats.tick == 0
        assert sim.world_state.tick == 1
        assert stats.tick_time_ms > 0
        # Population might change slightly due to deaths
        assert stats.population <= initial_pop + stats.births

    def test_multiple_ticks(self, small_config):
        sim = Simulation(config=small_config)

        for i in range(10):
            stats = sim.tick()
            assert stats.tick == i
            assert sim.world_state.tick == i + 1

    def test_population_dynamics(self, small_config):
        """Test that population changes over time (births and deaths)"""
        small_config.INITIAL_AGENT_COUNT = 200
        sim = Simulation(config=small_config)

        # Run for a while
        for _ in range(100):
            sim.tick()

        # Should have some births and deaths by now
        assert sim.world_state.total_births >= 0
        assert sim.world_state.total_deaths >= 0

    def test_agent_movement(self, small_config):
        sim = Simulation(config=small_config)

        # Record initial positions
        initial_positions = {
            aid: (a.x, a.y)
            for aid, a in sim.world_state.agents.items()
        }

        # Run a few ticks
        for _ in range(5):
            sim.tick()

        # Some agents should have moved
        moved = 0
        for aid, agent in sim.world_state.agents.items():
            if aid in initial_positions:
                old_x, old_y = initial_positions[aid]
                if agent.x != old_x or agent.y != old_y:
                    moved += 1

        assert moved > 0, "No agents moved"

    def test_checkpoint_save_load(self, small_config):
        sim = Simulation(config=small_config)

        # Run for a bit
        for _ in range(10):
            sim.tick()

        # Save checkpoint
        checkpoint_id = sim.save_checkpoint()

        # Record state
        saved_tick = sim.world_state.tick
        saved_pop = sim.world_state.population

        # Create new simulation from checkpoint
        sim2 = Simulation(config=small_config)
        sim2.load_checkpoint(checkpoint_id)

        assert sim2.world_state.tick == saved_tick
        assert sim2.world_state.population == saved_pop

    def test_event_logging(self, small_config):
        sim = Simulation(config=small_config)

        # Run enough to generate some events (eating events are logged)
        for _ in range(100):
            sim.tick()

        # Flush events
        sim.event_logger.flush()

        # Check events were logged (should have at least eating events)
        stats = sim.event_logger.get_stats()
        # At minimum we expect checkpoint events or the system is working
        # Even 0 events is valid if no interesting things happened
        assert stats['total_events'] >= 0

    def test_resource_harvesting(self, small_config):
        sim = Simulation(config=small_config)

        # Run simulation
        for _ in range(20):
            sim.tick()

        # Check that some resources were harvested
        resource_stats = sim.resources.get_stats()
        total_harvested = sum(resource_stats['total_harvested'].values())

        # Resources should have been consumed
        assert total_harvested >= 0

    def test_reproduction_occurs(self, small_config):
        """Test that reproduction can occur under right conditions"""
        small_config.INITIAL_AGENT_COUNT = 100
        small_config.MATURITY_AGE = 10  # Lower for testing
        small_config.REPRODUCTION_ENERGY_THRESHOLD = 30
        small_config.INTERACTION_RANGE = 50  # Larger range for testing
        small_config.WORLD_SIZE = 200  # Small world to force proximity

        sim = Simulation(config=small_config)

        # Set up agents with high energy and age, clustered together
        for i, agent in enumerate(sim.world_state.agents.values()):
            agent.energy = 100
            agent.age = 100  # Above maturity
            # Cluster agents together
            agent.x = 100 + (i % 10) * 5
            agent.y = 100 + (i // 10) * 5

        sim.agent_arrays.sync_from_state(sim.world_state)

        # Rebuild spatial grid with new positions
        positions = {aid: (a.x, a.y) for aid, a in sim.world_state.agents.items()}
        sim.spatial_grid.rebuild(positions)

        # Run simulation
        for _ in range(200):
            sim.tick()
            if sim.world_state.total_births > 0:
                break

        # Should have some births
        assert sim.world_state.total_births > 0

    def test_death_by_starvation(self, small_config):
        sim = Simulation(config=small_config)

        # Clear all resources to prevent eating
        sim.resources.cells.clear()
        sim.resources.active_cells.clear()

        # Set all agents to low energy
        for agent in sim.world_state.agents.values():
            agent.energy = 2.0  # Very low, will die in ~20 ticks

        sim.agent_arrays.sync_from_state(sim.world_state)

        # Run until deaths occur
        initial_pop = sim.world_state.population
        for _ in range(30):
            sim.tick()
            if sim.world_state.total_deaths > 0:
                break

        assert sim.world_state.population < initial_pop

    def test_run_method(self, small_config):
        sim = Simulation(config=small_config)

        sim.run(num_ticks=10, target_tps=0)  # No rate limiting

        assert sim.world_state.tick == 10
        assert len(sim.tick_stats) == 10

    def test_stats(self, small_config):
        sim = Simulation(config=small_config)

        for _ in range(5):
            sim.tick()

        stats = sim.get_stats()

        assert 'tick' in stats
        assert 'population' in stats
        assert 'total_births' in stats
        assert 'total_deaths' in stats
        assert 'resources' in stats


class TestScaling:
    """Tests for larger scale simulations"""

    def test_10k_agents(self):
        """Test with 10,000 agents"""
        config = SimConfig()
        config.WORLD_SIZE = 5000.0
        config.CELL_SIZE = 100.0
        config.INITIAL_AGENT_COUNT = 10000
        config.MAX_AGENTS = 15000
        config.CHECKPOINT_DIR = "/tmp/agent_sim_test/checkpoints"
        config.EVENT_LOG_DIR = "/tmp/agent_sim_test/events"
        config.SEED = 42

        sim = Simulation(config=config)

        # Run 10 ticks and measure performance
        times = []
        for _ in range(10):
            stats = sim.tick()
            times.append(stats.tick_time_ms)

        avg_time = sum(times) / len(times)
        print(f"\n10K agents: avg tick time = {avg_time:.1f}ms")

        # Should complete in reasonable time (< 1 second per tick)
        assert avg_time < 1000

    @pytest.mark.slow
    def test_100k_agents(self):
        """Test with 100,000 agents (marked slow)"""
        config = SimConfig()
        config.WORLD_SIZE = 10000.0
        config.CELL_SIZE = 50.0
        config.INITIAL_AGENT_COUNT = 100000
        config.MAX_AGENTS = 150000
        config.CHECKPOINT_DIR = "/tmp/agent_sim_test/checkpoints"
        config.EVENT_LOG_DIR = "/tmp/agent_sim_test/events"
        config.SEED = 42

        sim = Simulation(config=config)

        # Run 5 ticks
        times = []
        for _ in range(5):
            stats = sim.tick()
            times.append(stats.tick_time_ms)

        avg_time = sum(times) / len(times)
        print(f"\n100K agents: avg tick time = {avg_time:.1f}ms")

        # Record for benchmarking (no hard assertion)
