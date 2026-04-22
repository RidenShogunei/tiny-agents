"""Demo: Research Orchestrator — generate a survey paper.

Minimal end-to-end test of the full survey generation pipeline.

GPU usage (MultiGPUWriterPool):
    GPU 0: 2 writers + planner/reader/synthesizer (1.5B, shared KV cache)
    GPU 1: 2 writers (1.5B, shared KV cache)
    GPU 2: 1 writer + citation (0.5B)
"""

import asyncio
import os
import sys

sys.path.insert(0, "/home/jinxu/tiny-agents")

from tiny_agents.core.research_orchestrator import ResearchOrchestrator


def main():
    topic = "LoRA in Vision Models"

    print(f"{'=' * 60}")
    print(f"Research Orchestrator Demo")
    print(f"Topic: {topic}")
    print(f"{'=' * 60}")

    # Initialize orchestrator with 5 writers spread across 3 GPUs
    orchestrator = ResearchOrchestrator(
        num_writers=5,
        writer_model="Qwen/Qwen2.5-1.5B-Instruct",
        planner_model="Qwen/Qwen2.5-1.5B-Instruct",
        reader_model="Qwen/Qwen2.5-1.5B-Instruct",
        synthesizer_model="Qwen/Qwen2.5-1.5B-Instruct",
        citation_model="Qwen/Qwen2.5-0.5B-Instruct",
        max_papers=30,
        output_dir="/home/jinxu/tiny-agents/output",
    )

    # Set up VLLM backend
    # Set up VLLM backend
    # GPU layout (3 GPUs, each with one 1.5B model):
    #   GPU 0: 2 writers (shared 1.5B)
    #   GPU 1: 2 writers (shared 1.5B)
    #   GPU 2: 1 writer (1.5B)
    # Citation runs on CPU (轻量任务, 不占 GPU)
    from tiny_agents.models.vllm_backend import VLLMBackend
    backend = VLLMBackend(default_gpu=0)

    # GPU 0: 2 writers (shared 1.5B)
    backend.load_model("gpu0", "Qwen/Qwen2.5-1.5B-Instruct", gpu=0)
    # GPU 1: 2 writers (shared 1.5B)
    backend.load_model("gpu1", "Qwen/Qwen2.5-1.5B-Instruct", gpu=1)
    # GPU 2: 1 writer (1.5B)
    backend.load_model("gpu2", "Qwen/Qwen2.5-1.5B-Instruct", gpu=2)

    # Orchestrator's set_backend() detects GPU assignment from MultiGPUWriterPool
    # and loads remaining writer slots on the right GPUs
    orchestrator.set_backend(backend)

    # Run the pipeline
    result = asyncio.run(
        orchestrator.run(
            topic=topic,
            num_sections=6,
            max_papers=15,
        )
    )

    # Print results
    print()
    print(f"{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    if result["success"]:
        print(f"✓ Success!")
        print(f"  File: {result['filepath']}")
        print(f"  Sections: {result['sections_written']}")
        print(f"  Papers found: {result['papers_found']}")
        print(f"  Papers read: {result['papers_read']}")
        print(f"  Papers cited: {result['papers_cited']}")
        print(f"  Time: {result['execution_time']:.1f}s")
    else:
        print(f"✗ Failed: {result['errors']}")

    print()
    print("Step breakdown:")
    for step in result.get("steps", []):
        print(f"  {step['step']}: {step.get('time', 0):.1f}s")

    if result.get("errors"):
        print()
        print("Errors:")
        for e in result["errors"]:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
