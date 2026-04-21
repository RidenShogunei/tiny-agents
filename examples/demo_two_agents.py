"""Demo: Router + Coder two-agent collaboration."""

import asyncio
from tiny_agents.core.orchestrator import Orchestrator
from tiny_agents.agents.router import RouterAgent
from tiny_agents.agents.coder import CoderAgent


async def main():
    """Run a simple two-agent demo."""
    orch = Orchestrator()

    router = RouterAgent()
    coder = CoderAgent()

    orch.register_agent(router)
    orch.register_agent(coder)

    task = {
        "task": "Write a Python function that computes the nth Fibonacci number using memoization.",
    }

    result = await orch.execute(task, entry_agent="router")
    print("=" * 50)
    print("Execution Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
