"""Pipeline — generic multi-step processing pipeline.

A Pipeline is a directed acyclic graph of processing steps.
Each step runs an agent (or tool) with defined inputs/outputs.
Steps can run sequentially or in parallel (when dependencies allow).

Unlike Orchestrator (event-driven agent coordination), Pipeline is
data-flow oriented: define the steps upfront, then push data through.

Usage:
    from tiny_agents.core.pipeline import Pipeline, PipelineStep

    steps = [
        PipelineStep(id="plan", agent=planner, inputs=["topic"], outputs=["outline"]),
        PipelineStep(id="write", agent=writer, inputs=["outline"], outputs=["content"], parallel=True),
        PipelineStep(id="format", agent=formatter, inputs=["content"], outputs=["final"]),
    ]
    pipe = Pipeline(steps)
    result = await pipe.run({"topic": "LoRA in Vision Models"})

Step types:
    - agent   — runs BaseAgent.run() with input_data
    - tool    — runs BaseTool.execute() with input_data
    - inline  — runs a async callable directly
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)


class StepRunner(Protocol):
    """Protocol for anything that can run a pipeline step."""

    async def run_step(self, step_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        ...


@dataclass
class PipelineStep:
    """Definition of one step in a pipeline."""

    id: str
    # What to run: an agent, a tool, or an async callable
    runner: Union[Any, Callable]  # BaseAgent | BaseTool | Callable
    # Which other steps' outputs to use as inputs
    depends_on: List[str] = field(default_factory=list)
    # Input keys to extract from pipeline context (and pass to runner)
    input_keys: List[str] = field(default_factory=list)
    # Output keys to store back into pipeline context
    output_keys: List[str] = field(default_factory=list)
    # If True, all steps with parallel=True that have their dependencies met run concurrently
    parallel: bool = False
    # Extra config passed to runner
    config: Dict[str, Any] = field(default_factory=dict)

    def _is_agent(self) -> bool:
        from tiny_agents.core.agent import BaseAgent
        return isinstance(self.runner, BaseAgent)

    def _is_tool(self) -> bool:
        from tiny_agents.tools.base import BaseTool
        return isinstance(self.runner, BaseTool)

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this step with the given input data."""
        step_input = {k: input_data.get(k) for k in self.input_keys if k in input_data}

        # Allow runner to also read from context (shared state)
        step_input["_context"] = context

        if self._is_agent():
            from tiny_agents.core.session import SessionContext
            session = SessionContext(session_id=f"pipe_{self.id}", config=self.config)
            session._tools = context.get("_tools")
            output = await self.runner.run(step_input, session)
            result = output.payload if hasattr(output, "payload") else {"result": output}
        elif self._is_tool():
            result = self.runner.execute(step_input)
            if hasattr(result, "output"):
                result = result.output or {}
            elif not isinstance(result, dict):
                result = {"result": result}
        elif callable(self.runner):
            result = await self.runner(step_input, context)
        else:
            raise ValueError(f"Step {self.id}: runner type {type(self.runner)} not supported")

        return result


@dataclass
class PipelineResult:
    """Result of a pipeline run."""
    success: bool
    outputs: Dict[str, Any]  # step_id -> output dict
    errors: Dict[str, str]    # step_id -> error message
    final_output: Optional[Dict[str, Any]] = None


class Pipeline:
    """Generic multi-step pipeline executor.

    Resolves dependencies and executes steps in topological order.
    Steps marked `parallel=True` run concurrently when dependencies are met.
    """

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps
        self._step_map: Dict[str, PipelineStep] = {s.id: s for s in steps}

    def _resolve_order(self) -> List[List[str]]:
        """Resolve step execution order. Returns list of parallel groups."""
        in_degree: Dict[str, int] = {s.id: len(s.depends_on) for s in self.steps}
        dependents: Dict[str, List[str]] = {s.id: [] for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                if dep in dependents:
                    dependents[dep].append(s.id)

        groups: List[List[str]] = []
        remaining = set(self._step_map.keys())
        pending_parallels: List[str] = []

        while remaining:
            # Find all steps with in_degree == 0
            ready = [sid for sid in remaining if in_degree[sid] == 0]

            if not ready:
                # Circular dependency or bug — break
                logger.error(f"Deadlock: remaining={remaining}, in_degree={in_degree}")
                break

            # Separate parallel-friendly steps from sequential-only
            parallel_ready = [sid for sid in ready if self._step_map[sid].parallel]
            seq_ready = [sid for sid in ready if not self._step_map[sid].parallel]

            if seq_ready:
                groups.append(seq_ready)
            if parallel_ready:
                groups.append(parallel_ready)

            # Remove ready steps and update in_degrees
            for sid in ready:
                remaining.discard(sid)
                for dependent in dependents[sid]:
                    if dependent in in_degree:
                        in_degree[dependent] -= 1

        return groups

    async def run(self, initial_input: Dict[str, Any]) -> PipelineResult:
        """Execute the pipeline with the given input data."""
        context: Dict[str, Any] = dict(initial_input)
        outputs: Dict[str, Any] = {}
        errors: Dict[str, str] = {}

        groups = self._resolve_order()

        for group in groups:
            # All steps in a group are independent — run concurrently
            async def run_one(step_id: str) -> tuple:
                step = self._step_map[step_id]
                try:
                    # Gather dependencies' outputs
                    dep_data: Dict[str, Any] = {}
                    for dep_id in step.depends_on:
                        if dep_id in outputs:
                            dep_data.update(outputs[dep_id])

                    # Merge with context
                    step_input = {**context, **dep_data}

                    logger.info(f"[Pipeline] Running step: {step_id}")
                    result = await step.execute(step_input, context)

                    # Store selected outputs back into context
                    for key in step.output_keys:
                        if key in result:
                            context[key] = result[key]
                        elif key == "_all":
                            context[step_id] = result

                    outputs[step_id] = result
                    return step_id, result, None
                except Exception as e:
                    logger.exception(f"[Pipeline] Step {step_id} failed: {e}")
                    errors[step_id] = str(e)
                    return step_id, None, str(e)

            results = await asyncio.gather(
                *[run_one(sid) for sid in group],
                return_exceptions=True,
            )

            for res in results:
                if isinstance(res, Exception):
                    errors["unknown"] = str(res)

        final = None
        if outputs:
            last_step_id = list(outputs.keys())[-1]
            final = outputs.get(last_step_id)

        return PipelineResult(
            success=len(errors) == 0,
            outputs=outputs,
            errors=errors,
            final_output=final,
        )
