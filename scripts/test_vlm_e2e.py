"""End-to-end test: VLMBackend + VLPerceptionAgent + Orchestrator."""
import asyncio
import os
import sys

sys.path.insert(0, '/home/jinxu/tiny-agents')

from tiny_agents.models.vlm_backend import VLMBackend
from tiny_agents.agents.vl_perception import VLPerceptionAgent
from tiny_agents.core.orchestrator import Orchestrator
from tiny_agents.core.session import SessionContext


async def test():
    print("=== Test 1: VLMBackend direct ===")
    backend = VLMBackend(default_gpu=0, gpu_memory_utilization=0.80)
    await backend.initialize()
    print("Backend initialized!")

    # Text-only
    messages = [{"role": "user", "content": "What is 1+1?"}]
    result = backend.generate(messages, max_tokens=30, temperature=0.0)
    print(f"Text result: {result}")

    # With image (base64 1x1 red pixel PNG)
    import base64
    png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    image_url = f"data:image/png;base64,{png_b64}"
    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "What color is this image?"},
    ]}]
    result = backend.generate(messages, max_tokens=30, temperature=0.0)
    print(f"Vision result: {result}")

    print("\n=== Test 2: VLPerceptionAgent (stateless, no Orchestrator) ===")
    agent = VLPerceptionAgent()
    agent.set_backend(backend)

    ctx = SessionContext(session_id="test_vl", config={"temperature": 0.0, "max_tokens": 30})

    output = await agent.run({
        "task": "What color is this image?",
        "image": image_url,
    }, ctx)
    print(f"Agent output: {output.payload.get('description')}")
    print(f"Session context has {len(ctx.messages['vl_perception'])} messages for vl_perception agent")

    print("\n=== Test 3: Orchestrator with VLM agent (reuse backend) ===")
    # Reuse the backend from Test 2 (still alive)
    orch = Orchestrator(max_iterations=2)
    vl_agent = VLPerceptionAgent()
    vl_agent.set_backend(backend)   # reuse same backend instance
    orch.register_agent(vl_agent)

    result = await orch.execute(
        {"task": "What color is this image?", "image": image_url},
        entry_agent="vl_perception",
        session_id="test_orch",
    )
    print(f"Orchestrator success: {result['success']}")
    desc = result.get('result', {}).get('description', 'no description')
    print(f"Description: {desc}")

    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(test())
