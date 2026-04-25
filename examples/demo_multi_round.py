"""Demo: Router -> Coder -> Critic multi-round collaboration.

This demo shows:
    1. Router decomposes task
    2. Coder writes code
    3. Critic reviews code
    4. If issues found, Coder rewrites with feedback
    5. Full trace stored in SharedMemory
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tiny_agents.models.vllm_backend import VLLMBackend
from tiny_agents.core.orchestrator import Orchestrator
from tiny_agents.agents.router import RouterAgent
from tiny_agents.agents.coder import CoderAgent
from tiny_agents.agents.critic import CriticAgent


async def main():
    print("=" * 70)
    print("Tiny Agents Demo: Multi-Round Coder-Critic Loop")
    print("=" * 70)

    backend = VLLMBackend(default_gpu=0)

    # Load models across GPUs
    backend.load_model("router", "Qwen/Qwen2.5-1.5B-Instruct", gpu=0, max_model_len=4096)
    backend.load_model("coder", "Qwen/Qwen2.5-3B-Instruct", gpu=1, max_model_len=4096)
    backend.load_model("critic", "Qwen/Qwen2.5-0.5B-Instruct", gpu=2, max_model_len=4096)

    # Create agents
    router = RouterAgent(backend=backend)
    coder = CoderAgent(backend=backend)
    critic = CriticAgent(backend=backend)

    # Orchestrator with review chain enabled
    orch = Orchestrator(max_iterations=3, enable_review=True)
    orch.register_agent(router)
    orch.register_agent(coder)
    orch.register_agent(critic)
    orch.critic_agent = "critic"

    # Task: write a function with a deliberate edge-case bug for critic to catch
    task = {
        "task": (
            "Write a Python function `average(numbers)` that computes the average of a list. "
            "Intentionally forget to handle the empty list edge case so the critic can catch it."
        ),
    }

    print(f"\n[Task] {task['task']}\n")

    # Execute with multi-round support
    result = await orch.execute(task, entry_agent="router")

    print("\n" + "=" * 70)
    print("Execution Trace")
    print("=" * 70)
    for step in result.get("steps", []):
        agent = step["agent"]
        out = step["output"]
        print(f"\n[Step {step['step']}] Agent: {agent}")
        print(f"  Thought: {out['thought']}")
        print(f"  Action:  {out['action']}")
        if out.get("payload"):
            payload = out["payload"]
            if "code" in payload:
                print(f"  Code:\n{payload['code'][:500]}...")
            if "review" in payload:
                print(f"  Review:\n{payload['review'][:300]}...")
            if "verdict" in payload:
                print(f"  Verdict: {payload['verdict']}")

    print("\n" + "=" * 70)
    print("Final Result")
    print("=" * 70)
    if result.get("success"):
        print("SUCCESS")
        if result["result"].get("code"):
            print(f"\n{result['result']['code']}\n")
    else:
        print(f"FAILED: {result.get('reason')}")

    # Show memory trace
    print("\n" + "=" * 70)
    print("SharedMemory Session Context")
    print("=" * 70)
    session_id = result.get("session_id", "unknown")
    ctx = orch.memory.get_session_context(session_id)
    print(f"Total memory entries for session: {len(ctx)}")

    backend.unload_all()
    print("\n[Demo] Done. Models unloaded.")


if __name__ == "__main__":
    asyncio.run(main())
