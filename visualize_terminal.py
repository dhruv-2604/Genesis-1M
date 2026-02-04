"""Terminal-based ASCII visualization"""

import os
import sys
import time
from src.core.simulation_llm import LLMSimulation
from src.core.state import AgentTier, GeneIndex
from src.config import SimConfig


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def render_world(sim: LLMSimulation, width: int = 60, height: int = 30) -> str:
    """Render world to ASCII grid"""
    # Create empty grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]

    world_size = sim.config.WORLD_SIZE
    scale_x = width / world_size
    scale_y = height / world_size

    # Place agents
    alive_indices = sim.agent_arrays.get_alive_indices()

    for idx in alive_indices:
        x = int(sim.agent_arrays.x[idx] * scale_x)
        y = int(sim.agent_arrays.y[idx] * scale_y)

        # Clamp to grid
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        tier = sim.agent_arrays.tier[idx]
        genes = sim.agent_arrays.genes[idx]

        # Choose character based on tier and personality
        if tier == AgentTier.TIER1:
            char = '★'  # LLM-driven
        elif genes[GeneIndex.AGGRESSION] > 0.7:
            char = '!'  # Aggressive
        elif genes[GeneIndex.SOCIABILITY] > 0.7:
            char = 'o'  # Social
        else:
            char = '·'  # Normal

        grid[y][x] = char

    # Build string
    border = '+' + '-' * width + '+'
    lines = [border]
    for row in grid:
        lines.append('|' + ''.join(row) + '|')
    lines.append(border)

    return '\n'.join(lines)


def render_stats(sim: LLMSimulation) -> str:
    """Render statistics"""
    tick = sim.world_state.tick
    pop = sim.world_state.population
    t1_stats = sim.tier1_processor.get_stats()

    return f"""
╔══════════════════════════════════════╗
║  SIMULATION STATUS                   ║
╠══════════════════════════════════════╣
║  Tick: {tick:>10,}                   ║
║  Population: {pop:>7,}                ║
║                                      ║
║  Tier 1 (LLM): {t1_stats['active_tier1']:>5}                ║
║  Promotions: {t1_stats['total_promotions']:>7,}              ║
║  Memories: {t1_stats['total_memories_created']:>7,}                ║
╠══════════════════════════════════════╣
║  Legend:                             ║
║    ★ = LLM-driven (Tier 1)           ║
║    ! = Aggressive                    ║
║    o = Social                        ║
║    · = Normal                        ║
╚══════════════════════════════════════╝
"""


def main():
    """Run terminal visualization"""
    config = SimConfig()
    config.INITIAL_AGENT_COUNT = 200
    config.WORLD_SIZE = 300
    config.LOG_INTERVAL = 0
    config.CHECKPOINT_INTERVAL = 0
    config.MATURITY_AGE = 100
    config.MAX_AGE = 2000
    config.BASE_ENERGY_DRAIN = 0.3

    print("Initializing simulation...")
    sim = LLMSimulation(config=config, use_mock_llm=True)
    print(f"Created {sim.world_state.population} agents")
    print("Starting visualization (Ctrl+C to stop)...")
    time.sleep(1)

    try:
        while True:
            # Run some ticks
            for _ in range(5):
                sim.tick()

            # Render
            clear_screen()
            print(render_world(sim))
            print(render_stats(sim))

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
