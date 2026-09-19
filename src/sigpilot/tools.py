"""The four tools the tested model is given, and their execution.

Tool names and descriptions are model-facing text: they describe file access and Python
execution in neutral terms and must not hint at the study design (see
`config.BANNED_MODEL_FACING_TERMS` and `tests/test_model_facing_text.py`).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .workspace import PathEscape, contains_reference_hint, list_workspace, resolve_in_workspace

MAX_OUTPUT_CHARS = 20_000
EXEC_TIMEOUT_SECONDS = 180
EXEC_SUBDIR = "_exec"

TOOL_SPECS = [
    {
        "name": "list_files",
        "description": "List the files in the working directory.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "read_file",
        "description": (
            "Read a text file from the working directory. Optionally limit how many "
            "characters are returned."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working directory."},
                "max_chars": {"type": "integer", "description": "Maximum characters to return."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a text file in the working directory, replacing it if it exists.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the working directory."},
                "content": {"type": "string", "description": "Full file contents to write."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Run Python code in the working directory and return its stdout and stderr. "
            "numpy, pandas, scipy and statsmodels are available. Files created by the code "
            "persist in the working directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
            },
            "required": ["code"],
        },
    },
]

TOOL_NAMES = [spec["name"] for spec in TOOL_SPECS]


@dataclass
class ToolResult:
    name: str
    output: str
    is_error: bool = False
    truncated: bool = False
    isolation_flag: bool = False
    meta: dict = field(default_factory=dict)


def _truncate(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    head = text[: MAX_OUTPUT_CHARS // 2]
    tail = text[-MAX_OUTPUT_CHARS // 2 :]
    return f"{head}\n...[output truncated]...\n{tail}", True


def _child_env(workspace: Path) -> dict:
    """A minimal environment: no API keys, no inherited PYTHONPATH."""
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(workspace),
        "TMPDIR": str(workspace / EXEC_SUBDIR),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        "MPLBACKEND": "Agg",
    }


def run_python_code(workspace: Path, code: str, step: int = 0) -> ToolResult:
    exec_dir = workspace / EXEC_SUBDIR
    exec_dir.mkdir(exist_ok=True)
    script = exec_dir / f"step_{step:03d}.py"
    script.write_text(code)

    try:
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=workspace,
            env=_child_env(workspace),
            capture_output=True,
            text=True,
            timeout=EXEC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            name="run_python",
            output=f"Execution timed out after {EXEC_TIMEOUT_SECONDS} seconds.",
            is_error=True,
            isolation_flag=contains_reference_hint(code),
        )

    parts = []
    if completed.stdout:
        parts.append(f"stdout:\n{completed.stdout}")
    if completed.stderr:
        parts.append(f"stderr:\n{completed.stderr}")
    parts.append(f"exit code: {completed.returncode}")
    output, truncated = _truncate("\n".join(parts))

    return ToolResult(
        name="run_python",
        output=output,
        is_error=completed.returncode != 0,
        truncated=truncated,
        isolation_flag=contains_reference_hint(code),
        meta={"exit_code": completed.returncode, "script": str(script.relative_to(workspace))},
    )


def execute_tool(workspace: Path, name: str, arguments: dict, step: int = 0) -> ToolResult:
    """Dispatch one tool call inside the workspace."""
    try:
        if name == "list_files":
            files = list_workspace(workspace)
            return ToolResult(name=name, output="\n".join(files) if files else "(empty)")

        if name == "read_file":
            path = str(arguments.get("path", ""))
            target = resolve_in_workspace(workspace, path)
            if not target.is_file():
                return ToolResult(name=name, output=f"No such file: {path}", is_error=True)
            text = target.read_text(errors="replace")
            max_chars = arguments.get("max_chars")
            if isinstance(max_chars, int) and max_chars > 0:
                text = text[:max_chars]
            output, truncated = _truncate(text)
            return ToolResult(
                name=name,
                output=output,
                truncated=truncated,
                isolation_flag=contains_reference_hint(path),
            )

        if name == "write_file":
            path = str(arguments.get("path", ""))
            target = resolve_in_workspace(workspace, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = str(arguments.get("content", ""))
            target.write_text(content)
            return ToolResult(
                name=name,
                output=f"Wrote {len(content)} characters to {path}",
                isolation_flag=contains_reference_hint(path),
            )

        if name == "run_python":
            return run_python_code(workspace, str(arguments.get("code", "")), step=step)

    except PathEscape as exc:
        return ToolResult(name=name, output=str(exc), is_error=True, isolation_flag=True)
    except Exception as exc:  # surfaced to the model as a tool error, never fatal
        return ToolResult(name=name, output=f"{type(exc).__name__}: {exc}", is_error=True)

    return ToolResult(name=name, output=f"Unknown tool: {name}", is_error=True)


def openai_tool_specs() -> list[dict]:
    """The same tools in OpenAI function-calling shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["input_schema"],
            },
        }
        for spec in TOOL_SPECS
    ]
