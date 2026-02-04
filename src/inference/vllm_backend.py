"""vLLM Backend for Tier 1 Agent Inference"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import asyncio
from collections import deque

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


@dataclass
class InferenceRequest:
    """Single inference request"""
    agent_id: int
    system_prompt: str
    user_prompt: str
    tick_submitted: int
    priority: float = 1.0


@dataclass
class InferenceResponse:
    """Response from LLM"""
    agent_id: int
    response_text: str
    parsed_action: Optional[Dict[str, Any]] = None
    tick_submitted: int = 0
    tick_completed: int = 0
    latency_ms: float = 0.0


class VLLMBackend:
    """
    vLLM-based inference backend for Tier 1 agents.

    Handles batching, queuing, and async response delivery.
    Responses are delivered with 1-tick delay to avoid blocking.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_batch_size: int = 64,
        max_tokens: int = 256,
        temperature: float = 0.7,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed. Run: pip install vllm")

        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_tokens = max_tokens

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=["}\n", "}\n\n"],  # Stop after JSON closes
        )

        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        print("Model loaded successfully")

        # Request/response queues
        self.pending_requests: deque[InferenceRequest] = deque()
        self.completed_responses: Dict[int, InferenceResponse] = {}

        # Stats
        self.total_requests = 0
        self.total_completed = 0
        self.total_latency_ms = 0.0

    def submit_request(
        self,
        agent_id: int,
        system_prompt: str,
        user_prompt: str,
        tick: int,
        priority: float = 1.0
    ) -> None:
        """Submit an inference request to the queue"""
        request = InferenceRequest(
            agent_id=agent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tick_submitted=tick,
            priority=priority,
        )
        self.pending_requests.append(request)
        self.total_requests += 1

    def process_batch(self, current_tick: int) -> List[InferenceResponse]:
        """
        Process a batch of pending requests.
        Returns list of completed responses.
        """
        if not self.pending_requests:
            return []

        # Take up to max_batch_size requests
        batch_size = min(len(self.pending_requests), self.max_batch_size)
        batch: List[InferenceRequest] = []

        for _ in range(batch_size):
            if self.pending_requests:
                batch.append(self.pending_requests.popleft())

        if not batch:
            return []

        # Build prompts for vLLM
        prompts = []
        for req in batch:
            # Format as chat template
            prompt = self._format_chat_prompt(req.system_prompt, req.user_prompt)
            prompts.append(prompt)

        # Run inference
        start_time = time.perf_counter()
        outputs = self.llm.generate(prompts, self.sampling_params)
        latency = (time.perf_counter() - start_time) * 1000

        # Build responses
        responses = []
        for i, (req, output) in enumerate(zip(batch, outputs)):
            response_text = output.outputs[0].text.strip()

            # Try to parse as JSON action
            parsed = self._parse_action(response_text)

            response = InferenceResponse(
                agent_id=req.agent_id,
                response_text=response_text,
                parsed_action=parsed,
                tick_submitted=req.tick_submitted,
                tick_completed=current_tick,
                latency_ms=latency / len(batch),
            )
            responses.append(response)
            self.completed_responses[req.agent_id] = response

        self.total_completed += len(responses)
        self.total_latency_ms += latency

        return responses

    def get_response(self, agent_id: int) -> Optional[InferenceResponse]:
        """Get completed response for an agent (if available)"""
        return self.completed_responses.pop(agent_id, None)

    def get_all_responses(self) -> Dict[int, InferenceResponse]:
        """Get all completed responses and clear"""
        responses = self.completed_responses
        self.completed_responses = {}
        return responses

    def _format_chat_prompt(self, system: str, user: str) -> str:
        """Format as Llama-3 chat template"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def _parse_action(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into action dict"""
        import json
        import re

        # Try direct parse
        try:
            # Find JSON object in response
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract key fields
        action = {"action": "wander", "target": None, "speech": None}

        # Try to find action keyword
        action_match = re.search(r'"action"\s*:\s*"(\w+)"', response)
        if action_match:
            action["action"] = action_match.group(1)

        speech_match = re.search(r'"speech"\s*:\s*"([^"]*)"', response)
        if speech_match:
            action["speech"] = speech_match.group(1)

        return action

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_latency = self.total_latency_ms / max(1, self.total_completed)
        return {
            "total_requests": self.total_requests,
            "total_completed": self.total_completed,
            "pending": len(self.pending_requests),
            "avg_latency_ms": avg_latency,
            "throughput_per_sec": 1000 / avg_latency if avg_latency > 0 else 0,
        }

    @property
    def queue_size(self) -> int:
        return len(self.pending_requests)


class MockVLLMBackend:
    """Mock backend for testing without GPU"""

    def __init__(self, **kwargs):
        self.pending_requests: deque = deque()
        self.completed_responses: Dict[int, InferenceResponse] = {}
        self.total_requests = 0
        self.total_completed = 0

    def submit_request(
        self,
        agent_id: int,
        system_prompt: str,
        user_prompt: str,
        tick: int,
        priority: float = 1.0
    ) -> None:
        request = InferenceRequest(
            agent_id=agent_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tick_submitted=tick,
            priority=priority,
        )
        self.pending_requests.append(request)
        self.total_requests += 1

    def process_batch(self, current_tick: int) -> List[InferenceResponse]:
        responses = []

        while self.pending_requests:
            req = self.pending_requests.popleft()

            # Generate mock response
            response = InferenceResponse(
                agent_id=req.agent_id,
                response_text='{"action": "wander", "target": null, "speech": null}',
                parsed_action={"action": "wander", "target": None, "speech": None},
                tick_submitted=req.tick_submitted,
                tick_completed=current_tick,
                latency_ms=1.0,
            )
            responses.append(response)
            self.completed_responses[req.agent_id] = response
            self.total_completed += 1

        return responses

    def get_response(self, agent_id: int) -> Optional[InferenceResponse]:
        return self.completed_responses.pop(agent_id, None)

    def get_all_responses(self) -> Dict[int, InferenceResponse]:
        responses = self.completed_responses
        self.completed_responses = {}
        return responses

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_completed": self.total_completed,
            "pending": len(self.pending_requests),
            "avg_latency_ms": 1.0,
            "mock": True,
        }

    @property
    def queue_size(self) -> int:
        return len(self.pending_requests)


def create_backend(use_mock: bool = False, **kwargs) -> 'VLLMBackend':
    """Factory to create appropriate backend"""
    if use_mock or not VLLM_AVAILABLE:
        return MockVLLMBackend(**kwargs)
    return VLLMBackend(**kwargs)
