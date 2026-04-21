"""Demo: Router + Coder two-agent collaboration with real vLLM inference."""

import asyncio
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_agents.models.vllm_backend import VLLMBackend
from tiny_agents.core.orchestrator import Orchestrator
from tiny_agents.agents.router import RouterAgent
from tiny_agents.agents.coder import CoderAgent


async def main():
    """Run a simple two-agent demo with real models."""
    print("=" * 60)
    print("Tiny Agents Demo: Router -> Coder")
    print("=" * 60)

    # Initialize inference backend
    backend = VLLMBackend(default_gpu=0)

    # Load models (will take a moment on first run)
    # Router uses 1.5B for fast decision-making
    backend.load_model(
        model_key="router",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_model_len=4096,
    )

    # Coder uses 3B for better code generation
    backend.load_model(
        model_key="coder",
        model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
        max_model_len=4096,
    )

    # Create agents with backend
    router = RouterAgent(backend=backend)
    coder = CoderAgent(backend=backend)

    # Set up orchestrator
    orch = Orchestrator()
    orch.register_agent(router)
    orch.register_agent(coder)

    # Demo task
    task = {
        "task": "Write a Python function that computes the nth Fibonacci number using memoization.",
    }

    print(f"\n[Task] {task['task']}\n")

    result = await orch.execute(task, entry_agent="router")

    print("\n" + "=" * 60)
    print("Execution Result:")
    print("=" * 60)
    if result.get("success"):
        code = result["result"].get("code", "No code generated")
        print(f"\n{code}\n")
    else:
        print(f"Failed: {result.get('reason', 'unknown')}")
        for step in result.get("steps", []):
            print(f"  Step {step['step']}: {step['agent']} -> {step['output']['action']}")

    # Cleanup
    backend.unload_all()
    print("[Demo] Done. Models unloaded.")


if __name__ == "__main__":
    asyncio.run(main())
