"""Simple visualization for the simulation"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque

from src.core.simulation_llm import LLMSimulation
from src.core.state import AgentTier, GeneIndex
from src.config import SimConfig


class SimulationVisualizer:
    """Real-time visualization of the simulation"""

    def __init__(self, sim: LLMSimulation, update_interval: int = 5):
        self.sim = sim
        self.update_interval = update_interval  # Ticks between visual updates

        # Event log for displaying recent actions
        self.event_log = deque(maxlen=10)

        # Setup figure
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
        self.ax_world = self.axes[0]
        self.ax_stats = self.axes[1]

        # History for stats
        self.pop_history = deque(maxlen=200)
        self.tick_history = deque(maxlen=200)

        self._setup_world_view()
        self._setup_stats_view()

    def _setup_world_view(self):
        """Setup the world visualization"""
        self.ax_world.set_xlim(0, self.sim.config.WORLD_SIZE)
        self.ax_world.set_ylim(0, self.sim.config.WORLD_SIZE)
        self.ax_world.set_aspect('equal')
        self.ax_world.set_title('World View')
        self.ax_world.set_facecolor('#2d5016')  # Dark green background

        # Scatter plots for different agent types
        self.scatter_t3 = self.ax_world.scatter([], [], s=8, c='white', alpha=0.5, label='Tier 3')
        self.scatter_t1 = self.ax_world.scatter([], [], s=30, c='yellow', marker='*', label='Tier 1 (LLM)')

        # Legend
        self.ax_world.legend(loc='upper right', fontsize=8)

    def _setup_stats_view(self):
        """Setup the statistics panel"""
        self.ax_stats.set_xlim(0, 200)
        self.ax_stats.set_ylim(0, 300)
        self.ax_stats.set_title('Statistics & Events')
        self.ax_stats.axis('off')

    def _get_agent_colors(self, indices):
        """Color agents by dominant personality trait"""
        colors = []
        for idx in indices:
            genes = self.sim.agent_arrays.genes[idx]

            # Find dominant personality trait
            agg = genes[GeneIndex.AGGRESSION]
            soc = genes[GeneIndex.SOCIABILITY]
            alt = genes[GeneIndex.ALTRUISM] if GeneIndex.ALTRUISM < len(genes) else 0.5

            if agg > 0.7:
                colors.append('red')  # Aggressive
            elif alt > 0.7:
                colors.append('blue')  # Altruistic
            elif soc > 0.7:
                colors.append('green')  # Social
            else:
                colors.append('white')  # Neutral

        return colors

    def update(self, frame):
        """Update visualization"""
        # Run simulation ticks
        for _ in range(self.update_interval):
            stats = self.sim.tick()

        tick = self.sim.world_state.tick
        pop = self.sim.world_state.population

        # Update history
        self.tick_history.append(tick)
        self.pop_history.append(pop)

        # Get alive agents
        alive_mask = self.sim.agent_arrays.get_alive_mask()
        alive_indices = np.where(alive_mask[:self.sim.agent_arrays.count])[0]

        if len(alive_indices) == 0:
            return

        # Separate by tier
        t1_mask = self.sim.agent_arrays.tier[alive_indices] == AgentTier.TIER1
        t3_mask = ~t1_mask

        t1_indices = alive_indices[t1_mask]
        t3_indices = alive_indices[t3_mask]

        # Update Tier 3 scatter
        if len(t3_indices) > 0:
            x_t3 = self.sim.agent_arrays.x[t3_indices]
            y_t3 = self.sim.agent_arrays.y[t3_indices]
            colors_t3 = self._get_agent_colors(t3_indices)
            self.scatter_t3.set_offsets(np.c_[x_t3, y_t3])
            self.scatter_t3.set_facecolors(colors_t3)
        else:
            self.scatter_t3.set_offsets(np.empty((0, 2)))

        # Update Tier 1 scatter
        if len(t1_indices) > 0:
            x_t1 = self.sim.agent_arrays.x[t1_indices]
            y_t1 = self.sim.agent_arrays.y[t1_indices]
            self.scatter_t1.set_offsets(np.c_[x_t1, y_t1])
        else:
            self.scatter_t1.set_offsets(np.empty((0, 2)))

        # Update title
        self.ax_world.set_title(f'World View - Tick {tick:,} | Pop: {pop:,}')

        # Update stats panel
        self.ax_stats.clear()
        self.ax_stats.axis('off')
        self.ax_stats.set_title('Statistics')

        # Stats text
        tier1_stats = self.sim.tier1_processor.get_stats()
        stats_text = f"""
Tick: {tick:,}
Population: {pop:,}

Tier 1 (LLM): {tier1_stats['active_tier1']}
Promotions: {tier1_stats['total_promotions']:,}
Demotions: {tier1_stats['total_demotions']:,}

Memories: {tier1_stats['total_memories_created']:,}

Color Legend:
  Red = Aggressive
  Blue = Altruistic
  Green = Social
  White = Neutral
  ★ = Tier 1 (LLM-driven)
"""
        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace')

        return self.scatter_t3, self.scatter_t1

    def run(self, num_frames=1000, interval=100):
        """Run the visualization"""
        ani = FuncAnimation(self.fig, self.update, frames=num_frames,
                           interval=interval, blit=False)
        plt.tight_layout()
        plt.show()


def main():
    """Run visualization demo"""
    # Small config for visualization
    config = SimConfig()
    config.INITIAL_AGENT_COUNT = 300
    config.WORLD_SIZE = 400
    config.LOG_INTERVAL = 0
    config.CHECKPOINT_INTERVAL = 0

    # Faster lifecycle
    config.MATURITY_AGE = 100
    config.MAX_AGE = 2000
    config.BASE_ENERGY_DRAIN = 0.3
    config.REPRODUCTION_COOLDOWN = 50

    print("Initializing simulation...")
    sim = LLMSimulation(config=config, use_mock_llm=True)
    print(f"Created {sim.world_state.population} agents")

    print("Starting visualization (close window to stop)...")
    viz = SimulationVisualizer(sim, update_interval=3)
    viz.run(num_frames=10000, interval=50)


if __name__ == "__main__":
    main()
