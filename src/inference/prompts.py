"""LLM Prompt Templates for Tier 1 Agents

Philosophy: The LLM is the "hardware" (reasoning engine), not the "database" (knowledge).
We use prompts to:
1. Inject personality from genes (nature)
2. Block future knowledge (they only know what they've discovered)
3. Force reasoning with available resources only
"""

from typing import Dict, List, Any, Optional
import json
import re


def _personality_description(genes: Dict[str, float]) -> str:
    """Convert gene values to natural language personality description"""
    traits = []

    # Aggression (0 = pacifist, 1 = violent)
    agg = genes.get('aggression', 0.5)
    if agg > 0.7:
        traits.append("You are aggressive and quick to anger. Violence feels natural to you.")
    elif agg > 0.5:
        traits.append("You will fight if provoked, but prefer to avoid conflict.")
    elif agg < 0.3:
        traits.append("You are peaceful and avoid violence at almost any cost.")

    # Sociability (0 = loner, 1 = social)
    soc = genes.get('sociability', 0.5)
    if soc > 0.7:
        traits.append("You crave company and feel uneasy when alone.")
    elif soc < 0.3:
        traits.append("You prefer solitude and find others exhausting.")

    # Altruism (0 = selfish, 1 = self-sacrificing)
    alt = genes.get('altruism', 0.5)
    if alt > 0.7:
        traits.append("You instinctively help others, even at cost to yourself.")
    elif alt < 0.3:
        traits.append("You look out for yourself first. Others' problems are not yours.")

    # Greed (0 = generous, 1 = hoards)
    greed = genes.get('greed', 0.5)
    if greed > 0.7:
        traits.append("You hate sharing. What's yours is yours.")
    elif greed < 0.3:
        traits.append("You share freely. Resources are meant to flow.")

    # Curiosity (0 = conservative, 1 = explorer)
    cur = genes.get('curiosity', 0.5)
    if cur > 0.7:
        traits.append("You are driven to explore and try new things.")
    elif cur < 0.3:
        traits.append("You prefer familiar routines and distrust novelty.")

    # Trust (0 = paranoid, 1 = naive)
    trust = genes.get('trust', 0.5)
    if trust > 0.7:
        traits.append("You tend to believe what others tell you.")
    elif trust < 0.3:
        traits.append("You are suspicious. Everyone has an agenda.")

    return " ".join(traits) if traits else "You have a balanced temperament."


# System prompt - establishes the "empty hands" constraint
SYSTEM_PROMPT_TEMPLATE = """You are {name}, a primitive human in an ancient world.

CRITICAL RULES:
1. You have NO knowledge of the future. No guns, no metal tools, no farming unless you've discovered it.
2. You only know what you have PERSONALLY seen, done, or been told by others.
3. You can only use items you are CURRENTLY holding. You cannot imagine items into existence.
4. Your reasoning is intelligent, but your knowledge is limited to direct experience.

YOUR PERSONALITY (this is who you ARE, not who you choose to be):
{personality}

SURVIVAL PRIORITIES:
- Find food when hungry
- Find safety when threatened
- Find companions when lonely (if social)
- Explore when curious (if curious)

You think step by step, but your thoughts are shaped by your personality and limited knowledge."""


def build_system_prompt(agent_state: Dict[str, Any]) -> str:
    """Build system prompt with personality injection"""
    name = agent_state.get('name', f"Human-{agent_state['id']}")

    # Extract personality genes
    genes = {}
    if 'genes' in agent_state:
        gene_array = agent_state['genes']
        genes = {
            'aggression': float(gene_array[5]) if len(gene_array) > 5 else 0.5,
            'sociability': float(gene_array[6]) if len(gene_array) > 6 else 0.5,
            'altruism': float(gene_array[7]) if len(gene_array) > 7 else 0.5,
            'greed': float(gene_array[8]) if len(gene_array) > 8 else 0.5,
            'curiosity': float(gene_array[9]) if len(gene_array) > 9 else 0.5,
            'trust': float(gene_array[10]) if len(gene_array) > 10 else 0.5,
        }

    personality = _personality_description(genes)

    return SYSTEM_PROMPT_TEMPLATE.format(
        name=name,
        personality=personality,
    )


# Action prompt - the "what do you do" question
ACTION_PROMPT_TEMPLATE = """CURRENT SITUATION:

Your body:
- Energy: {energy}/100 {energy_status}
- You have lived {age} seasons
- You feel: {physical_state}

What you are holding:
{inventory}

What you see RIGHT NOW:
{observations}

What you remember:
{memories}

IMPORTANT: You can ONLY use what you are holding. You cannot wish for tools you don't have.

What do you do? First, think step by step about your situation. Then respond with JSON:
{{
  "thought": "Your internal reasoning (what you're thinking, based on your personality)",
  "action": "move|eat|rest|approach|flee|attack|offer_trade|mate|gather|craft|speak",
  "target": "direction/agent_name/resource/item or null",
  "speech": "What you say out loud (or null if silent)"
}}"""


def build_prompt(
    agent_state: Dict[str, Any],
    nearby_agents: List[Dict[str, Any]],
    nearby_resources: Dict[str, float],
    terrain_type: str,
    memories: List[str],
) -> str:
    """Build action prompt for Tier 1 agent"""

    energy = int(agent_state.get('energy', 100))

    # Energy status
    if energy < 15:
        energy_status = "⚠️ DYING - you must eat NOW or die!"
        physical_state = "weak, dizzy, desperate"
    elif energy < 30:
        energy_status = "(very hungry)"
        physical_state = "hungry, tired, irritable"
    elif energy < 50:
        energy_status = "(somewhat hungry)"
        physical_state = "alert but could use food"
    elif energy > 80:
        energy_status = "(well-fed)"
        physical_state = "strong, satisfied, confident"
    else:
        energy_status = ""
        physical_state = "normal"

    # Inventory - what they're HOLDING (critical constraint)
    inventory = agent_state.get('inventory', {})
    if inventory:
        inv_lines = [f"- {v}x {k}" for k, v in inventory.items()]
        inv_str = "\n".join(inv_lines)
    else:
        inv_str = "- Nothing. Your hands are empty."

    # Observations - what they can SEE
    observations = []

    # Terrain
    terrain_desc = {
        'PLAINS': "Open grassland. Easy to move, easy to be seen.",
        'FOREST': "Dense trees. Good cover, hard to navigate.",
        'MOUNTAIN': "Rocky slopes. Difficult terrain, caves possible.",
        'WATER': "Water blocks your path. Cannot cross without swimming.",
        'DESERT': "Dry wasteland. No food, no water visible.",
    }
    observations.append(f"Terrain: {terrain_desc.get(terrain_type, 'Unknown terrain')}")

    # Resources
    for resource, amount in nearby_resources.items():
        if amount > 0:
            if amount > 5:
                observations.append(f"- Plenty of {resource} nearby (easy to gather)")
            elif amount > 2:
                observations.append(f"- Some {resource} visible")
            else:
                observations.append(f"- A small amount of {resource} (scarce)")

    # Other agents
    if nearby_agents:
        for agent in nearby_agents[:5]:
            name = agent.get('name', f"Human-{agent['id']}")
            e = agent.get('energy', 50)

            if e < 20:
                condition = "weak and starving"
            elif e > 80:
                condition = "healthy and strong"
            else:
                condition = "appears normal"

            observations.append(f"- {name}: {condition}")
    else:
        observations.append("- No other humans visible. You are alone.")

    # Memories
    if memories:
        memory_str = "\n".join(f"- {m}" for m in memories[-5:])
    else:
        memory_str = "- Nothing notable has happened recently."

    return ACTION_PROMPT_TEMPLATE.format(
        energy=energy,
        energy_status=energy_status,
        age=agent_state.get('age', 0) // 365,
        physical_state=physical_state,
        inventory=inv_str,
        observations="\n".join(observations),
        memories=memory_str,
    )


def parse_action_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into action dict"""

    # Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from response
    json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "action": "move",
        "target": "random",
        "speech": None,
        "thought": "confused"
    }


# Crafting prompt - for when agent tries to craft
CRAFT_PROMPT_TEMPLATE = """You are trying to make something useful.

You are holding:
{inventory}

You want to: {goal}

RULES:
- You can ONLY use items you are holding
- You cannot use items you don't have
- Think about how these items could physically combine

What can you make with these items? If nothing useful, say "nothing".

Respond with JSON:
{{
  "can_craft": true/false,
  "result": "name of item you can make (or null)",
  "reasoning": "how you would combine the items"
}}"""


def build_craft_prompt(inventory: Dict[str, int], goal: str) -> str:
    """Build prompt for crafting attempt"""
    if inventory:
        inv_str = "\n".join(f"- {v}x {k}" for k, v in inventory.items())
    else:
        inv_str = "Nothing. Your hands are empty."

    return CRAFT_PROMPT_TEMPLATE.format(inventory=inv_str, goal=goal)


# Social prompt - for complex social situations
SOCIAL_PROMPT_TEMPLATE = """You encounter {other_name}.

Your relationship: {relationship}
Your personality: {personality}
Their apparent state: {other_state}

{situation}

How do you react? Consider your personality - you can't act against your nature.

Respond with JSON:
{{
  "feeling": "how you feel about this person",
  "action": "what you do",
  "speech": "what you say (or null)"
}}"""


def build_social_prompt(
    agent_genes: Dict[str, float],
    other_name: str,
    relationship: str,
    other_state: str,
    situation: str,
) -> str:
    """Build prompt for social interaction"""
    personality = _personality_description(agent_genes)

    return SOCIAL_PROMPT_TEMPLATE.format(
        other_name=other_name,
        relationship=relationship,
        personality=personality,
        other_state=other_state,
        situation=situation,
    )


# Memory generation prompt
MEMORY_PROMPT = """Summarize what happened in ONE sentence from {name}'s perspective.
Keep it simple - this is a primitive human's memory.

What happened: {event}

Memory (max 12 words):"""


def build_memory_prompt(agent_name: str, event_description: str) -> str:
    """Build prompt for generating memory summary"""
    return MEMORY_PROMPT.format(name=agent_name, event=event_description)
