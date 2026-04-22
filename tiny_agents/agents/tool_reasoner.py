"""Tool-augmented reasoner agent with Python code execution."""

import re
from typing import Any, Dict

from tiny_agents.core.agent import BaseAgent, AgentOutput
from tiny_agents.core.session import SessionContext
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
    """Reasoner that can execute Python code to verify calculations (stateless)."""

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
        # Try ```python ... ```
        for pattern in [
            r"```python\n(.*?)\n```",
            r"```python\n(.*)",
            r"```\n(.*?)\n```",
            r"```\n(.*)",
        ]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""

    def _extract_answer(self, text: str) -> str:
        """Extract numerical answer from model output."""
        if "####" in text:
            parts = text.split("####")
            last = parts[-1].strip().split()[0] if parts[-1].strip() else ""
            return last.replace(",", "")
        match = re.search(r"\\boxed\{(.*?)\}", text)
        if match:
            return match.group(1).replace(",", "")
        lines = text.strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return line.replace(",", "")
        return text.strip().replace(",", "")

    async def run(
        self,
        input_data: Dict[str, Any],
        context: SessionContext,
    ) -> AgentOutput:
        """Solve problem with optional Python tool use (stateless)."""
        task = input_data.get("task", "")

        if self.backend is None:
            return AgentOutput(
                thought="No backend available",
                action="respond",
                payload={"answer": "ERROR: no backend"},
                finished=True,
            )

        # Build fresh messages from context
        messages = context.get_messages(self.name)
        messages.append({"role": "user", "content": task})

        temp = context.config.get("temperature", 0.1)
        max_tok = context.config.get("max_tokens", 1024)

        # Step 1: Generate initial reasoning (may include Python code)
        response1 = self.backend.generate(
            model_key=self.model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tok,
        )
        context.add_message(self.name, "assistant", response1)

        # Step 2: Check if Python code was generated
        code = self._extract_python_code(response1)
        if code:
            exec_result = self.executor.run(code)
            tool_output = (
                f"Python execution result:\n"
                f"stdout: {exec_result['stdout']}\n"
                f"stderr: {exec_result['stderr']}\n"
            )
            context.add_message(self.name, "user", tool_output)

            # Step 3: Generate final answer with tool output
            response2 = self.backend.generate(
                model_key=self.model_name,
                messages=context.get_messages(self.name),
                temperature=temp,
                max_tokens=256,
            )
            context.add_message(self.name, "assistant", response2)
            final_answer = self._extract_answer(response2)
        else:
            final_answer = self._extract_answer(response1)

        return AgentOutput(
            thought="Solved with tool-augmented reasoning" + (" (used Python)" if code else ""),
            action="respond",
            payload={"answer": final_answer, "code": code},
            finished=True,
        )
