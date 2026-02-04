"""LLM Prompt Templates for Tier 1 Agents (Phase 2)"""

from typing import Dict, List, Any, Optional


# System prompt for primitive human agents
PRIMITIVE_HUMAN_SYSTEM = """You are a primitive human named '{name}'. You have high logical reasoning capabilities, but you have NO knowledge of modern technology, history, or science.

Your only goals are:
1. Find food
2. Find shelter
3. Survive

If you meet another human, you can choose to cooperate or fight.

You experience the world through simple observations and must make decisions based on immediate survival needs. You can form basic social bonds, share resources, and work together - or compete for scarce resources.

Respond ONLY with a JSON action. Do not explain your reasoning."""


# Action prompt template
ACTION_PROMPT = """Current state:
- Energy: {energy}/100 {energy_status}
- Age: {age} seasons
- Location: {terrain_type} terrain

You are carrying: {inventory}

You see around you:
{observations}

Recent memories:
{memories}

What do you do? Respond with JSON:
{{
  "action": "move|eat|rest|approach|flee|offer_trade|attack|mate|gather|craft",
  "target": "direction/agent_name/resource_name or null",
  "speech": "What you say out loud (or null)",
  "thought": "Your internal reasoning (1 sentence)"
}}"""


# Observation templates
def format_agent_observation(agent_data: Dict[str, Any], relationship: str = "stranger") -> str:
    """Format another agent for observation"""
    name = agent_data.get('name', f"Human-{agent_data['id']}")
    state = "resting" if agent_data.get('fsm_state') == 4 else "moving"

    if agent_data.get('energy', 100) < 20:
        condition = "looks weak and hungry"
    elif agent_data.get('energy', 100) > 80:
        condition = "looks healthy and strong"
    else:
        condition = "looks normal"

    return f"- {name} ({relationship}): {state}, {condition}"


def format_resource_observation(resource_type: str, amount: float, distance: str) -> str:
    """Format a resource for observation"""
    quantity = "abundant" if amount > 5 else "some" if amount > 2 else "scarce"
    return f"- {quantity} {resource_type} ({distance})"


def format_terrain_observation(terrain_type: str) -> str:
    """Format terrain for observation"""
    descriptions = {
        'PLAINS': "open grassland stretches before you",
        'FOREST': "dense trees surround you, offering cover",
        'MOUNTAIN': "rocky terrain rises steeply",
        'WATER': "a body of water blocks your path",
        'DESERT': "dry, barren land with little vegetation",
    }
    return descriptions.get(terrain_type, "unfamiliar terrain")


def build_prompt(
    agent_state: Dict[str, Any],
    nearby_agents: List[Dict[str, Any]],
    nearby_resources: Dict[str, float],
    terrain_type: str,
    memories: List[str],
) -> str:
    """Build full prompt for Tier 1 agent decision"""

    # Format observations
    observations = []

    # Terrain
    observations.append(f"Terrain: {format_terrain_observation(terrain_type)}")

    # Resources
    for resource, amount in nearby_resources.items():
        if amount > 0:
            observations.append(format_resource_observation(resource, amount, "nearby"))

    # Other agents
    for agent in nearby_agents[:5]:  # Limit to 5 nearest
        observations.append(format_agent_observation(agent))

    if not nearby_agents:
        observations.append("- No other humans in sight")

    # Format inventory
    inventory = agent_state.get('inventory', {})
    if inventory:
        inv_str = ", ".join(f"{v} {k}" for k, v in inventory.items())
    else:
        inv_str = "nothing"

    # Format memories
    if memories:
        memory_str = "\n".join(f"- {m}" for m in memories[-3:])
    else:
        memory_str = "- (no recent memories)"

    # Energy status text
    energy = int(agent_state.get('energy', 100))
    if energy < 20:
        energy_status = "(CRITICAL - you are starving!)"
    elif energy < 40:
        energy_status = "(hungry)"
    else:
        energy_status = ""

    return ACTION_PROMPT.format(
        energy=energy,
        energy_status=energy_status,
        age=agent_state.get('age', 0) // 365,  # Convert ticks to "seasons"
        terrain_type=terrain_type,
        inventory=inv_str,
        observations="\n".join(observations),
        memories=memory_str,
    )


def build_system_prompt(agent_state: Dict[str, Any]) -> str:
    """Build system prompt for agent"""
    name = agent_state.get('name', f"Human-{agent_state['id']}")
    return PRIMITIVE_HUMAN_SYSTEM.format(name=name)


# Response parsing
def parse_action_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into action dict"""
    import json
    import re

    # Try to extract JSON from response
    try:
        # Direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in response
    json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: return wander action
    return {
        "action": "move",
        "target": "random",
        "speech": None,
        "thought": "confused"
    }


# Memory generation prompt
MEMORY_PROMPT = """Summarize this interaction in ONE short sentence from {name}'s perspective:

{event_description}

Summary (max 15 words):"""


def build_memory_prompt(agent_name: str, event_description: str) -> str:
    """Build prompt for generating memory summary"""
    return MEMORY_PROMPT.format(name=agent_name, event_description=event_description)
