"""Checkpointing system for save/restore simulation state"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import gzip
import time
import shutil
import hashlib

from .state import WorldState, AgentState


class CheckpointManager:
    """
    Manages saving and loading simulation checkpoints.

    Checkpoints are compressed JSON files containing full world state.
    Supports versioning, rollback, and cleanup of old checkpoints.
    """

    VERSION = 1  # Checkpoint format version

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 10,
        compress: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.compress = compress

    def save(
        self,
        world_state: WorldState,
        resource_state: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a checkpoint. Returns checkpoint ID.
        """
        tick = world_state.tick
        timestamp = int(time.time())
        checkpoint_id = f"checkpoint_{tick:012d}_{timestamp}"

        data = {
            'version': self.VERSION,
            'checkpoint_id': checkpoint_id,
            'tick': tick,
            'timestamp': timestamp,
            'world_state': world_state.to_dict(),
            'resource_state': resource_state,
            'metadata': metadata or {},
        }

        # Compute checksum for integrity verification
        data_str = json.dumps(data, sort_keys=True)
        checksum = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        data['checksum'] = checksum

        # Write checkpoint
        if self.compress:
            filepath = self.checkpoint_dir / f"{checkpoint_id}.json.gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
        else:
            filepath = self.checkpoint_dir / f"{checkpoint_id}.json"
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f)

        # Cleanup old checkpoints
        self._cleanup()

        return checkpoint_id

    def load(self, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load a checkpoint by ID. If no ID provided, load latest.
        Returns dict with 'world_state', 'resource_state', 'metadata'.
        """
        if checkpoint_id is None:
            checkpoint_id = self._get_latest_id()
            if checkpoint_id is None:
                raise FileNotFoundError("No checkpoints found")

        # Try compressed first, then uncompressed
        filepath = self.checkpoint_dir / f"{checkpoint_id}.json.gz"
        if not filepath.exists():
            filepath = self.checkpoint_dir / f"{checkpoint_id}.json"

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")

        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # Verify checksum
        stored_checksum = data.pop('checksum', None)
        if stored_checksum:
            data_str = json.dumps(data, sort_keys=True)
            computed_checksum = hashlib.sha256(data_str.encode()).hexdigest()[:16]
            if stored_checksum != computed_checksum:
                raise ValueError(f"Checkpoint checksum mismatch: {checkpoint_id}")

        # Convert world state
        world_state = WorldState.from_dict(data['world_state'])

        return {
            'world_state': world_state,
            'resource_state': data.get('resource_state'),
            'metadata': data.get('metadata', {}),
            'checkpoint_id': data['checkpoint_id'],
            'tick': data['tick'],
            'timestamp': data['timestamp'],
        }

    def load_latest(self) -> Dict[str, Any]:
        """Load the most recent checkpoint"""
        return self.load()

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata"""
        checkpoints = []

        for filepath in sorted(self.checkpoint_dir.glob("checkpoint_*.json*")):
            # Parse checkpoint ID from filename
            name = filepath.stem
            if name.endswith('.json'):
                name = name[:-5]

            parts = name.split('_')
            if len(parts) >= 3:
                try:
                    tick = int(parts[1])
                    timestamp = int(parts[2])
                    checkpoints.append({
                        'checkpoint_id': name,
                        'tick': tick,
                        'timestamp': timestamp,
                        'filepath': str(filepath),
                        'size_bytes': filepath.stat().st_size,
                    })
                except ValueError:
                    continue

        return sorted(checkpoints, key=lambda x: x['tick'], reverse=True)

    def delete(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        for ext in ['.json.gz', '.json']:
            filepath = self.checkpoint_dir / f"{checkpoint_id}{ext}"
            if filepath.exists():
                filepath.unlink()
                return True
        return False

    def _get_latest_id(self) -> Optional[str]:
        """Get the ID of the most recent checkpoint"""
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]['checkpoint_id']
        return None

    def _cleanup(self) -> None:
        """Remove old checkpoints beyond max_checkpoints"""
        checkpoints = self.list_checkpoints()

        if len(checkpoints) > self.max_checkpoints:
            for cp in checkpoints[self.max_checkpoints:]:
                self.delete(cp['checkpoint_id'])

    def get_checkpoint_at_tick(self, target_tick: int) -> Optional[Dict[str, Any]]:
        """Get the checkpoint closest to (but not after) a specific tick"""
        checkpoints = self.list_checkpoints()

        for cp in checkpoints:
            if cp['tick'] <= target_tick:
                return self.load(cp['checkpoint_id'])

        return None

    def verify_all(self) -> Dict[str, bool]:
        """Verify integrity of all checkpoints"""
        results = {}
        for cp in self.list_checkpoints():
            try:
                self.load(cp['checkpoint_id'])
                results[cp['checkpoint_id']] = True
            except Exception as e:
                results[cp['checkpoint_id']] = False
        return results


class IncrementalCheckpoint:
    """
    Incremental checkpointing that only saves changed agents.

    Useful for very large simulations where full checkpoints are expensive.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track which agents changed since last checkpoint
        self.changed_agents: set = set()
        self.last_full_tick: int = 0

    def mark_changed(self, agent_id: int) -> None:
        """Mark an agent as changed"""
        self.changed_agents.add(agent_id)

    def save_incremental(
        self,
        world_state: WorldState,
        base_checkpoint_id: str
    ) -> str:
        """Save only changed agents as a delta"""
        tick = world_state.tick
        timestamp = int(time.time())
        checkpoint_id = f"delta_{tick:012d}_{timestamp}"

        # Extract only changed agents
        changed_data = {}
        for agent_id in self.changed_agents:
            if agent_id in world_state.agents:
                changed_data[agent_id] = world_state.agents[agent_id].to_dict()
            else:
                changed_data[agent_id] = None  # Agent was deleted

        data = {
            'type': 'incremental',
            'base_checkpoint': base_checkpoint_id,
            'tick': tick,
            'timestamp': timestamp,
            'changed_agents': changed_data,
            'total_births': world_state.total_births,
            'total_deaths': world_state.total_deaths,
        }

        filepath = self.checkpoint_dir / f"{checkpoint_id}.json.gz"
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f)

        self.changed_agents.clear()
        return checkpoint_id

    def apply_delta(
        self,
        base_state: WorldState,
        delta_path: str
    ) -> WorldState:
        """Apply an incremental checkpoint to a base state"""
        with gzip.open(delta_path, 'rt', encoding='utf-8') as f:
            delta = json.load(f)

        # Apply changes
        for agent_id_str, agent_data in delta['changed_agents'].items():
            agent_id = int(agent_id_str)
            if agent_data is None:
                base_state.remove_agent(agent_id)
            else:
                agent = AgentState.from_dict(agent_data)
                base_state.agents[agent_id] = agent

        base_state.tick = delta['tick']
        base_state.total_births = delta['total_births']
        base_state.total_deaths = delta['total_deaths']

        return base_state
