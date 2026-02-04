"""RLM (Recursive Language Model) hook for Henri.

Usage:
    henri --hook hooks/rlm.py

Adds an rlm_query tool that uses RLM to analyze large contexts
(e.g., codebases) by recursively decomposing the problem with
sub-model calls inside a sandboxed Docker REPL.

Requires: Docker running, rlm package installed (pip install -e ../rlm)

Configure via environment variables:
    HENRI_RLM_BACKEND    - RLM backend (default: inferred from HENRI_PROVIDER)
    HENRI_RLM_MODEL      - RLM model (default: inferred from HENRI_MODEL)
    HENRI_RLM_MAX_ITERS  - Max RLM iterations (default: 15)
"""

import os
import tempfile

from henri.tools.base import Tool

# Map henri provider names to RLM backend names
PROVIDER_TO_BACKEND = {
    "bedrock": "bedrock",
    "google": "gemini",
    "vertex": "vertex",
    "ollama": "ollama",
    "openai_compatible": "vllm",
}


DEFAULT_MODELS = {
    "bedrock": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "gemini": "gemini-2.5-flash",
    "vertex": "claude-sonnet-4-5",
    "ollama": "qwen3-coder:30b",
}


def _get_rlm_config() -> dict:
    """Build RLM configuration from environment variables."""
    # Infer backend from henri's provider
    henri_provider = os.environ.get("HENRI_PROVIDER", "bedrock")
    backend = os.environ.get("HENRI_RLM_BACKEND", PROVIDER_TO_BACKEND.get(henri_provider, "bedrock"))

    # Infer model from henri's model, falling back to provider defaults
    model = os.environ.get("HENRI_RLM_MODEL") or os.environ.get("HENRI_MODEL") or DEFAULT_MODELS.get(backend)

    max_iters = int(os.environ.get("HENRI_RLM_MAX_ITERS", "15"))

    # Build backend_kwargs based on backend type
    backend_kwargs = {}
    if model:
        backend_kwargs["model_name"] = model

    if backend == "bedrock":
        region = os.environ.get("HENRI_REGION", "us-east-1")
        backend_kwargs["region"] = region
    elif backend == "vertex":
        region = os.environ.get("HENRI_REGION", "us-east5")
        backend_kwargs["region"] = region
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project:
            backend_kwargs["project"] = project
    elif backend == "ollama":
        host = os.environ.get("HENRI_HOST", "http://localhost:11434")
        backend_kwargs["host"] = host
    elif backend in ("vllm", "openai"):
        host = os.environ.get("HENRI_HOST")
        if host:
            backend_kwargs["base_url"] = f"{host}/v1"

    return {
        "backend": backend,
        "backend_kwargs": backend_kwargs,
        "max_iterations": max_iters,
    }


def _format_iterations(log_path: str) -> str:
    """Format RLM iterations from the log file for display."""
    import json

    output_parts = []
    try:
        with open(log_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("type") == "iteration":
                    i = entry.get("iteration", "?")
                    # Show code blocks and their results
                    for cb in entry.get("code_blocks", []):
                        code = cb.get("code", "").strip()
                        result = cb.get("result", {})
                        stdout = result.get("stdout", "").strip()
                        stderr = result.get("stderr", "").strip()

                        output_parts.append(f"--- Iteration {i} ---")
                        output_parts.append(f"Code:\n{code}")
                        if stdout:
                            output_parts.append(f"Output:\n{stdout}")
                        if stderr:
                            output_parts.append(f"Stderr:\n{stderr}")

                    if entry.get("final_answer"):
                        output_parts.append(f"--- Final Answer ---")
    except Exception as e:
        output_parts.append(f"[error reading log: {e}]")

    return "\n".join(output_parts)


class RLMQueryTool(Tool):
    """Analyze a codebase or large context using Recursive Language Models."""

    name = "rlm_query"
    description = (
        "Use RLM (Recursive Language Model) to analyze a codebase or large context. "
        "RLM spawns a sandboxed Docker REPL where it can read files, write Python code, "
        "and make sub-model calls to decompose the problem recursively. "
        "The project directory is mounted read-only at /project in the container. "
        "Returns a detailed analysis. Use this for tasks requiring deep codebase understanding "
        "across many files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The question or task to analyze (e.g., 'How is error handling done across this codebase?')",
            },
            "path": {
                "type": "string",
                "description": "Path to the project directory to analyze (default: current directory)",
                "default": ".",
            },
        },
        "required": ["query"],
    }
    requires_permission = True

    def execute(self, query: str, path: str = ".") -> str:
        try:
            from rlm import RLM
            from rlm.logger import RLMLogger
        except ImportError:
            return "[error: rlm package not installed. Install with: pip install -e path/to/rlm]"

        # Resolve project path
        project_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(project_path):
            return f"[error: directory not found: {path}]"

        # Get RLM config
        config = _get_rlm_config()

        # Set up logger to capture iterations
        log_dir = tempfile.mkdtemp(prefix="henri_rlm_")
        logger = RLMLogger(log_dir=log_dir, file_name="rlm")

        try:
            rlm = RLM(
                backend=config["backend"],
                backend_kwargs=config["backend_kwargs"],
                environment="docker",
                environment_kwargs={
                    "volumes": {project_path: "/project:ro"},
                },
                max_iterations=config["max_iterations"],
                max_depth=1,
                logger=logger,
                verbose=True,
            )

            # Build context prompt that tells RLM where to find the project
            context = (
                f"You have access to a project directory mounted at /project. "
                f"Use Python to explore it (os.walk, open, etc.). "
                f"The project is read-only.\n\n"
                f"Task: {query}"
            )

            result = rlm.completion(context, root_prompt=query)

            # Format output with iteration details
            iterations_log = _format_iterations(logger.log_file_path)
            output = ""
            if iterations_log:
                output += f"=== RLM Iterations ===\n{iterations_log}\n\n"
            output += f"=== RLM Result ===\n{result.response}"
            output += f"\n\n[RLM: {result.usage_summary.to_dict()}, time: {result.execution_time:.1f}s]"

            return output

        except Exception as e:
            return f"[error: RLM failed: {e}]"


# Tools to add
TOOLS = [RLMQueryTool()]

SYSTEM_PROMPT = (
    "You have access to the rlm_query tool which uses Recursive Language Models "
    "to analyze codebases. It spawns a sandboxed Docker container with the project "
    "mounted read-only at /project. The RLM can write Python code to explore files, "
    "search patterns, and call sub-models to analyze individual components. "
    "Use it when you need deep analysis across many files in a codebase."
)
