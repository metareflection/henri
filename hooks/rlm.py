"""RLM (Recursive Language Model) hook for Henri.

Usage:
    henri --hook hooks/rlm.py

Adds an rlm_query tool that uses RLM (https://github.com/alexzhang13/rlm)
to analyze large contexts (e.g., codebases) by recursively decomposing
the problem with sub-model calls inside a sandboxed local REPL.

The RLM session persists across calls as long as the project filesystem
hasn't changed. If any files have been modified, added, or deleted since
the last query, the session is automatically invalidated and a fresh one
is created. If a reused session times out, it retries with a fresh one.

We use a version of RLM customized for Henri:
    https://github.com/metareflection/rlm/tree/henri
which matches henri's providers and tweaks volume mounting. Install with:
    pip install -e ../rlm

Configure via environment variables:
    HENRI_RLM_BACKEND    - RLM backend (default: inferred from HENRI_PROVIDER)
    HENRI_RLM_MODEL      - RLM model (default: inferred from HENRI_MODEL)
    HENRI_RLM_MAX_ITERS  - Max RLM iterations (default: 15)
    HENRI_RLM_TIMEOUT    - Timeout per query in seconds (default: 60)
"""

import hashlib
import os
import signal
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


def _fs_fingerprint(path: str) -> str:
    """Hash file paths and mtimes under a directory to detect changes."""
    h = hashlib.md5()
    for root, dirs, files in sorted(os.walk(path)):
        # Skip hidden dirs (.git, .venv, etc.)
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for f in sorted(files):
            if f.startswith("."):
                continue
            fp = os.path.join(root, f)
            try:
                mtime = os.stat(fp).st_mtime_ns
            except OSError:
                continue
            h.update(f"{fp}\0{mtime}".encode())
    return h.hexdigest()


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
    """Analyze a codebase or large context using Recursive Language Models.

    The RLM session persists across calls as long as the project filesystem
    hasn't changed. If files are modified between calls, the session is
    automatically invalidated.
    """

    name = "rlm_query"
    description = (
        "Use RLM (Recursive Language Model) to analyze a codebase or large context. "
        "RLM uses a sandboxed REPL where it can read files, write Python code, "
        "and make sub-model calls to decompose the problem recursively. "
        "The session persists across calls as long as the project files haven't changed. "
        "If files were modified between calls, a fresh session is started automatically. "
        "Use this for tasks requiring deep codebase understanding across many files."
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

    def __init__(self):
        self._rlm = None
        self._project_path = None
        self._fs_fingerprint = None
        self._logger = None

    def _get_or_create_rlm(self, project_path: str) -> tuple:
        """Get existing RLM instance or create a new one if filesystem changed.

        Returns (rlm_instance, reused: bool).
        """
        from rlm import RLM
        from rlm.logger import RLMLogger

        fingerprint = _fs_fingerprint(project_path)

        # Reuse if same project and filesystem unchanged
        if (
            self._rlm is not None
            and self._project_path == project_path
            and self._fs_fingerprint == fingerprint
        ):
            return self._rlm, True

        # Clean up old instance
        if self._rlm is not None:
            self._rlm.close()

        config = _get_rlm_config()
        log_dir = tempfile.mkdtemp(prefix="henri_rlm_")
        self._logger = RLMLogger(log_dir=log_dir, file_name="rlm")

        self._rlm = RLM(
            backend=config["backend"],
            backend_kwargs=config["backend_kwargs"],
            environment="local",
            environment_kwargs={
                "setup_code": f"import os; PROJECT_DIR = {project_path!r}",
            },
            max_iterations=config["max_iterations"],
            max_depth=1,
            logger=self._logger,
            verbose=True,
            persistent=True,
        )
        self._project_path = project_path
        self._fs_fingerprint = fingerprint
        return self._rlm, False

    def execute(self, query: str, path: str = ".") -> str:
        try:
            from rlm import RLM
        except ImportError:
            return "[error: rlm package not installed. Install with: pip install -e path/to/rlm]"

        # Resolve project path
        project_path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isdir(project_path):
            return f"[error: directory not found: {path}]"

        timeout = int(os.environ.get("HENRI_RLM_TIMEOUT", "60"))

        try:
            result = self._run_with_timeout(project_path, query, timeout)
            return result
        except Exception as e:
            return f"[error: RLM failed: {e}]"

    def _run_query(self, project_path: str, query: str):
        """Run a single RLM query, returning (result, session_status)."""
        rlm, reused = self._get_or_create_rlm(project_path)
        session_status = "reused" if reused else "new"
        print(f"[RLM session: {session_status}]")

        context = (
            f"You have access to a project directory at PROJECT_DIR={project_path!r}. "
            f"Use Python to explore it (os.walk, open, etc.). "
            f"Variables from previous queries may be available in the REPL namespace.\n\n"
            f"Task: {query}"
        )

        result = rlm.completion(context, root_prompt=query)

        # Update fingerprint after successful completion
        self._fs_fingerprint = _fs_fingerprint(project_path)

        return result, session_status

    def _format_result(self, result, session_status: str) -> str:
        """Format an RLM result into output string."""
        iterations_log = _format_iterations(self._logger.log_file_path)
        output = f"[RLM session: {session_status}, time: {result.execution_time:.1f}s, {result.usage_summary.to_dict()}]\n\n"
        if iterations_log:
            output += f"=== RLM Iterations ===\n{iterations_log}\n\n"
        output += f"=== RLM Result ===\n{result.response}"
        return output

    def _run_with_timeout(self, project_path: str, query: str, timeout: int) -> str:
        """Run query with timeout. If a reused session times out, retry fresh."""
        was_reused = self._rlm is not None

        def _alarm_handler(signum, frame):
            raise TimeoutError()

        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        try:
            signal.alarm(timeout)
            result, session_status = self._run_query(project_path, query)
            signal.alarm(0)
            return self._format_result(result, session_status)
        except TimeoutError:
            signal.alarm(0)
            if was_reused:
                # Reused session timed out — retry with a fresh one
                print(f"[RLM session timed out after {timeout}s, retrying with fresh session]")
                if self._rlm is not None:
                    self._rlm.close()
                self._rlm = None
                self._fs_fingerprint = None
                signal.alarm(timeout)
                result, session_status = self._run_query(project_path, query)
                signal.alarm(0)
                return self._format_result(result, session_status)
            else:
                raise TimeoutError(f"RLM query timed out after {timeout}s")
        finally:
            signal.signal(signal.SIGALRM, old_handler)


# Tools to add
TOOLS = [RLMQueryTool()]

SYSTEM_PROMPT = (
    "You have access to the rlm_query tool which uses Recursive Language Models "
    "to analyze codebases. It uses a local REPL where the RLM can write Python code "
    "to explore files, search patterns, and call sub-models to analyze individual "
    "components. The session automatically persists across calls when the project "
    "files haven't changed, allowing iterative analysis. If files have been modified "
    "since the last query, a fresh session is started automatically. "
    "Use it when you need deep analysis across many files in a codebase."
)
