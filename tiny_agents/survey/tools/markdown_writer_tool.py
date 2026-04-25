"""MarkdownWriterTool — write or append Markdown content to files.

Provides structured file output for research reports, including automatic
directory creation and overwrite/append modes.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional

from tiny_agents.tools.base import BaseTool, ToolResult


class MarkdownWriterTool(BaseTool):
    """Write Markdown content to a file.

    Supports both overwrite and append modes. Automatically creates parent
    directories if they don't exist.

    Args:
        default_output_dir: Default directory for output files.
        default_filename: Default filename (supports {topic}, {date} placeholders).
    """

    def __init__(
        self,
        default_output_dir: str = "output",
        default_filename: Optional[str] = None,
    ):
        default_filename = default_filename or "{topic}_{date}.md"
        super().__init__(
            name="markdown_writer",
            description="Write or append Markdown content to a file.",
        )
        self.default_output_dir = default_output_dir
        self.default_filename = default_filename

    def _execute(
        self,
        content: str,
        path: str = None,
        filename: str = None,
        mode: str = "write",
    ) -> ToolResult:
        """Write Markdown content to file.

        Args:
            content: The Markdown content to write.
            path: Output directory. Defaults to self.default_output_dir.
            filename: Filename. Supports {topic}, {date} placeholders.
                      Defaults to self.default_filename.
            mode: 'write' (overwrite) or 'append'.
        """
        try:
            # Resolve output directory
            output_dir = path or self.default_output_dir
            output_dir = os.path.expanduser(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            # Resolve filename
            if filename:
                # Replace placeholders
                date_str = datetime.now().strftime("%Y%m%d")
                topic_placeholder = topic_from_content(content)
                filename = filename.replace("{topic}", topic_placeholder)
                filename = filename.replace("{date}", date_str)
            else:
                filename = self.default_filename.replace("{topic}", "report")
                filename = filename.replace("{date}", datetime.now().strftime("%Y%m%d"))

            # Ensure .md extension
            if not filename.endswith(".md"):
                filename += ".md"

            filepath = os.path.join(output_dir, filename)

            # Write or append
            if mode == "append":
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write("\n\n" + content)
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

            file_size = os.path.getsize(filepath)
            line_count = content.count("\n") + 1

            return ToolResult(
                tool_name=self.name,
                args={"content": f"<{len(content)} chars>", "path": filepath, "mode": mode},
                success=True,
                output={
                    "filepath": filepath,
                    "filename": filename,
                    "mode": mode,
                    "bytes_written": file_size,
                    "lines_written": line_count,
                },
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                args={"content": f"<{len(content)} chars>", "path": path, "mode": mode},
                success=False,
                error=f"Failed to write Markdown: {str(e)}",
            )

    def to_definition(self):
        """Return JSON Schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The Markdown content to write. Should be valid Markdown including headers, paragraphs, and references.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Output directory path. Defaults to 'output/'.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Filename template. Supports {topic} and {date} placeholders. Defaults to '{topic}_{date}.md'.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "'write' to overwrite, 'append' to add to existing file.",
                        "enum": ["write", "append"],
                        "default": "write",
                    },
                },
                "required": ["content"],
            },
        }


def topic_from_content(content: str) -> str:
    """Extract a topic-like string from content for filename generation."""
    # Try to find a title (# heading)
    import re
    heading = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if heading:
        title = heading.group(1).strip()
        # Sanitize for filename
        title = re.sub(r"[^\w\s\-]", "", title)
        title = re.sub(r"\s+", "_", title)[:40]
        return title
    return "report"
