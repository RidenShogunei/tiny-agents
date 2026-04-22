"""Tool-augmented reasoner agent with Python code execution."""

import re
from typing import Any, Dict

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.tools.python_executor import PythonExecutor


TOOL_REASONER_PROMPT = """You are a mathematical reasoning expert. You have access to a Python execution environment.

When solving problems:
1. Think step by step
2. If a calculation is complex or you are unsure, write Python code inside ```python ... ``` blocks to compute it
3. CRITICAL: Your Python code MUST end with a print() statement showing the result
4. After seeing the Python output, provide your final answer
5. End your final answer with the numerical result after ####

Example Python code format:
```python
x = 10 + 5
print(x)
```

Rules:
- Use Python for arithmetic, fractions, algebra verification, or any multi-step calculation
- Keep Python code simple and self-contained
- Always use print() to output the final computed value
- The final answer line must contain only #### followed by the number"""


class ToolReasonerAgent(BaseAgent):
    """Reasoner that can execute Python code to verify calculations."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", **kwargs):
        super().__init__(
            name="tool_reasoner",
            model_name=model_name,
            role_prompt=TOOL_REASONER_PROMPT,
            **kwargs,
        )
        self.executor = PythonExecutor(timeout=10)

    def _extract_python_code(self, text: str) -> str:
        """Extract first python code block, robust to missing closing tag."""
        # Try standard ```python ... ```
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try ```python ... (until end of text, no closing ```)
        pattern2 = r"```python\n(.*)"
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            return match2.group(1).strip()

        # Fallback: try without language tag
        pattern3 = r"```\n(.*?)\n```"
        match3 = re.search(pattern3, text, re.DOTALL)
        if match3:
            return match3.group(1).strip()

        pattern4 = r"```\n(.*)"
        match4 = re.search(pattern4, text, re.DOTALL)
        if match4:
            return match4.group(1).strip()

        return ""

    async def run(self, input_data: Dict[str, Any]) -> AgentOutput:
        """Solve problem with optional Python tool use."""
        task = input_data.get("task", "")

        if self.backend is None:
            return AgentOutput(
                thought="No backend available",
                action="respond",
                payload={"answer": "ERROR: no backend"},
                finished=True,
            )

        # Build fresh context for this problem (no accumulated history)
        messages = [
            {"role": "system", "content": self.role_prompt},
            {"role": "user", "content": task},
        ]

        # Step 1: Generate initial reasoning (may include Python code)
        response1 = self.backend.generate(
            model_key=self.name,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        messages.append({"role": "assistant", "content": response1})

        # Step 2: Check if Python code was generated
        code = self._extract_python_code(response1)
        if code:
            exec_result = self.executor.run(code)
            tool_output = (
                f"Python execution result:\n"
                f"stdout: {exec_result['stdout']}\n"
                f"stderr: {exec_result['stderr']}\n"
            )
            messages.append({"role": "user", "content": tool_output})

            # Step 3: Generate final answer with tool output
            response2 = self.backend.generate(
                model_key=self.name,
                messages=messages,
                temperature=0.1,
                max_tokens=256,
            )
            final_answer = response2
        else:
            final_answer = response1

        return AgentOutput(
            thought="Solved with tool-augmented reasoning" + (" (used Python)" if code else ""),
            action="respond",
            payload={"answer": final_answer, "code": code},
            finished=True,
        )
