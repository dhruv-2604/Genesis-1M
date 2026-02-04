"""Demo script to run the simulation with mock LLM"""

from src.core.simulation_llm import LLMSimulation
from src.config import SimConfig

# Configure a small simulation for demo
config = SimConfig()
config.INITIAL_AGENT_COUNT = 200      # Start with 200 agents
config.WORLD_SIZE = 500               # Smaller world
config.LOG_INTERVAL = 100             # Print every 100 ticks
config.CHECKPOINT_INTERVAL = 0        # Disable checkpoints for demo

# Speed up lifecycle for demo (normally much slower)
config.MATURITY_AGE = 100             # Can reproduce after 100 ticks
config.MAX_AGE = 2000                 # Die of old age at 2000 ticks
config.BASE_ENERGY_DRAIN = 0.5        # Faster energy drain
config.REPRODUCTION_COOLDOWN = 50     # Can reproduce more often

# Create simulation with mock LLM (no GPU needed)
sim = LLMSimulation(config=config, use_mock_llm=True)

print(f"Starting simulation with {sim.world_state.population} agents")
print(f"World size: {config.WORLD_SIZE}x{config.WORLD_SIZE}")
print("-" * 60)

# Run for 3000 ticks (enough for old age deaths at MAX_AGE=2000)
for i in range(3000):
    stats = sim.tick()

# Final summary
print("-" * 60)
print(f"\nFinal Stats after {sim.world_state.tick} ticks:")
print(f"  Population: {sim.world_state.population}")

tier1_stats = sim.tier1_processor.get_stats()
print(f"  Tier 1 agents: {tier1_stats['active_tier1']}")
print(f"  Total promotions: {tier1_stats['total_promotions']}")
print(f"  Total demotions: {tier1_stats['total_demotions']}")
print(f"  Memories created: {tier1_stats['total_memories_created']}")

sim_stats = sim.get_stats()
print(f"  Total births: {sim_stats.get('total_births', 0)}")
print(f"  Total deaths: {sim_stats.get('total_deaths', 0)}")

# Calculate avg tick time from tick_stats
if sim.tick_stats:
    avg_time = sum(s.tick_time_ms for s in sim.tick_stats) / len(sim.tick_stats)
    print(f"  Avg tick time: {avg_time:.1f}ms")
