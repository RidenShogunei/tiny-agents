"""Demo: Research Orchestrator — generate a survey paper.

Minimal end-to-end test of the full survey generation pipeline.

GPU layout:
    GPU 0: planner/reader/synthesizer/citation (1.5B, shared KV cache)
    GPU 1: 1.5B writers (shared)
    GPU 2: 1.5B writers (shared)
    GPU 3: 9B writer + 9B critic (Qwen3.5-9B, enforce_eager, gpu_mem=0.85)
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

    # Initialize orchestrator with 5 writers spread across 4 GPUs
    # GPU 3 = 9B (Qwen3.5-9B), GPU 0/1/2 = 1.5B
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
    from tiny_agents.models.vllm_backend import VLLMBackend
    backend = VLLMBackend(default_gpu=0)

    # GPU 0: planner, reader, synthesizer, citation (shared 1.5B)
    backend.load_model("gpu0", "Qwen/Qwen2.5-1.5B-Instruct", gpu=0)

    # GPU 1: writers (shared 1.5B)
    backend.load_model("gpu1", "Qwen/Qwen2.5-1.5B-Instruct", gpu=1)

    # GPU 2: writers (shared 1.5B)
    backend.load_model("gpu2", "Qwen/Qwen2.5-1.5B-Instruct", gpu=2)

    # GPU 3: 9B writer + critic (high quality writing)
    # enforce_eager=True avoids CUDA graph issues with large models
    # gpu_memory_utilization=0.85 leaves room for KV cache
    backend.load_model(
        "gpu3",
        "Qwen/Qwen3.5-9B",
        gpu=3,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
    )

    # Orchestrator detects GPU assignment and distributes writers
    # Writers on GPU 3 use 9B for higher quality output
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
