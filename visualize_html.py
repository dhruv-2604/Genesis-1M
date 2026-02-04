"""Generate HTML visualization snapshots"""

import json
import os
from pathlib import Path
from src.core.simulation_llm import LLMSimulation
from src.core.state import AgentTier, GeneIndex
from src.config import SimConfig


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Agent Simulation - Tick {tick}</title>
    <style>
        body {{
            font-family: 'Monaco', 'Consolas', monospace;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            display: flex;
            gap: 20px;
        }}
        #world {{
            background: #16213e;
            border: 2px solid #0f3460;
            border-radius: 8px;
        }}
        .stats {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            min-width: 300px;
        }}
        .stats h2 {{
            color: #e94560;
            margin-top: 0;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #0f3460;
        }}
        .legend {{
            margin-top: 20px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 5px 0;
        }}
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}
        .events {{
            margin-top: 20px;
            max-height: 200px;
            overflow-y: auto;
        }}
        .event {{
            padding: 5px;
            margin: 2px 0;
            background: #0f3460;
            border-radius: 4px;
            font-size: 12px;
        }}
        h1 {{
            color: #e94560;
        }}
    </style>
</head>
<body>
    <h1>Agent Simulation Snapshot</h1>
    <div class="container">
        <canvas id="world" width="{canvas_width}" height="{canvas_height}"></canvas>
        <div class="stats">
            <h2>Statistics</h2>
            <div class="stat-row"><span>Tick:</span><span>{tick:,}</span></div>
            <div class="stat-row"><span>Population:</span><span>{population:,}</span></div>
            <div class="stat-row"><span>Tier 1 (LLM):</span><span>{tier1_count}</span></div>
            <div class="stat-row"><span>Promotions:</span><span>{promotions:,}</span></div>
            <div class="stat-row"><span>Memories:</span><span>{memories:,}</span></div>

            <div class="legend">
                <h3>Legend</h3>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #ffd700;"></div>
                    <span>★ Tier 1 (LLM-driven)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #ff4444;"></div>
                    <span>Aggressive</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #4444ff;"></div>
                    <span>Altruistic</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #44ff44;"></div>
                    <span>Social</span>
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: #888888;"></div>
                    <span>Neutral</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const agents = {agents_json};
        const worldSize = {world_size};

        const canvas = document.getElementById('world');
        const ctx = canvas.getContext('2d');
        const scale = canvas.width / worldSize;

        // Draw agents
        agents.forEach(agent => {{
            const x = agent.x * scale;
            const y = agent.y * scale;

            ctx.beginPath();

            if (agent.tier === 1) {{
                // Tier 1 - star shape
                ctx.fillStyle = '#ffd700';
                drawStar(ctx, x, y, 5, 8, 4);
            }} else {{
                // Color by personality
                if (agent.aggression > 0.7) {{
                    ctx.fillStyle = '#ff4444';
                }} else if (agent.altruism > 0.7) {{
                    ctx.fillStyle = '#4444ff';
                }} else if (agent.sociability > 0.7) {{
                    ctx.fillStyle = '#44ff44';
                }} else {{
                    ctx.fillStyle = '#888888';
                }}
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.fill();
            }}
        }});

        function drawStar(ctx, cx, cy, spikes, outerRadius, innerRadius) {{
            let rot = Math.PI / 2 * 3;
            let x = cx;
            let y = cy;
            const step = Math.PI / spikes;

            ctx.beginPath();
            ctx.moveTo(cx, cy - outerRadius);

            for (let i = 0; i < spikes; i++) {{
                x = cx + Math.cos(rot) * outerRadius;
                y = cy + Math.sin(rot) * outerRadius;
                ctx.lineTo(x, y);
                rot += step;

                x = cx + Math.cos(rot) * innerRadius;
                y = cy + Math.sin(rot) * innerRadius;
                ctx.lineTo(x, y);
                rot += step;
            }}

            ctx.lineTo(cx, cy - outerRadius);
            ctx.closePath();
            ctx.fill();
        }}
    </script>
</body>
</html>
"""


def generate_snapshot(sim: LLMSimulation, output_path: str = "snapshot.html"):
    """Generate HTML snapshot of current simulation state"""

    # Collect agent data
    agents = []
    alive_indices = sim.agent_arrays.get_alive_indices()

    for idx in alive_indices:
        genes = sim.agent_arrays.genes[idx]
        agents.append({
            'x': float(sim.agent_arrays.x[idx]),
            'y': float(sim.agent_arrays.y[idx]),
            'energy': float(sim.agent_arrays.energy[idx]),
            'tier': int(sim.agent_arrays.tier[idx]),
            'aggression': float(genes[GeneIndex.AGGRESSION]),
            'sociability': float(genes[GeneIndex.SOCIABILITY]),
            'altruism': float(genes[GeneIndex.ALTRUISM]) if GeneIndex.ALTRUISM < len(genes) else 0.5,
        })

    t1_stats = sim.tier1_processor.get_stats()

    # Generate HTML
    html = HTML_TEMPLATE.format(
        tick=sim.world_state.tick,
        population=sim.world_state.population,
        tier1_count=t1_stats['active_tier1'],
        promotions=t1_stats['total_promotions'],
        memories=t1_stats['total_memories_created'],
        canvas_width=600,
        canvas_height=600,
        world_size=sim.config.WORLD_SIZE,
        agents_json=json.dumps(agents),
    )

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    """Generate snapshot after running simulation"""
    config = SimConfig()
    config.INITIAL_AGENT_COUNT = 500
    config.WORLD_SIZE = 500
    config.LOG_INTERVAL = 100
    config.CHECKPOINT_INTERVAL = 0
    config.MATURITY_AGE = 100
    config.MAX_AGE = 2000
    config.BASE_ENERGY_DRAIN = 0.3

    print("Initializing simulation...")
    sim = LLMSimulation(config=config, use_mock_llm=True)

    print(f"Running simulation for 500 ticks...")
    for i in range(500):
        sim.tick()

    print("Generating HTML snapshot...")
    path = generate_snapshot(sim, "snapshot.html")
    print(f"Snapshot saved to: {path}")
    print(f"Open in browser: file://{os.path.abspath(path)}")


if __name__ == "__main__":
    main()
