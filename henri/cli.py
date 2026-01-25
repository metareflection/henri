"""CLI entry point for Henri."""

import argparse
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

from henri.agent import run_agent
from henri.config import (
    DEFAULT_PROVIDER,
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_GOOGLE_MODEL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_VERTEX_MODEL,
)
from henri.providers import PROVIDERS


def env_var(name: str, default=None):
    """Get environment variable with HENRI_ prefix."""
    return os.environ.get(f"HENRI_{name}", default)


def load_hook(hook_path: str):
    """Load a hook module from a file path."""
    path = Path(hook_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Hook file not found: {hook_path}")

    # Use unique module name based on file path to avoid collisions
    module_name = f"hook_{path.stem}_{id(path)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Henri - A pedagogical Claude Code clone. "
        "All options can be set via HENRI_<OPTION> env vars (e.g., HENRI_PROVIDER).",
    )
    parser.add_argument(
        "--provider", "-p",
        choices=list(PROVIDERS.keys()),
        default=env_var("PROVIDER", DEFAULT_PROVIDER),
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model", "-m",
        default=env_var("MODEL"),
        help="Model ID (provider-specific, uses default if not set)",
    )
    parser.add_argument(
        "--region",
        default=env_var("REGION"),
        help="Region (Bedrock: AWS region, Vertex: GCP region)",
    )
    parser.add_argument(
        "--host",
        default=env_var("HOST"),
        help="Host URL for Ollama or OpenAI-compatible providers",
    )
    parser.add_argument(
        "--hook",
        action="append",
        help="Path to a Python hook file (can be used multiple times)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum conversation turns (default: unlimited)",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default=env_var("STATS_FILE"),
        help="Path to write JSON stats (turns, tokens) after completion",
    )
    args = parser.parse_args()

    # Apply env vars for special cases (int conversion, list splitting)
    if args.max_turns is None:
        max_turns_env = env_var("MAX_TURNS")
        if max_turns_env:
            args.max_turns = int(max_turns_env)

    if args.hook is None:
        hook_env = env_var("HOOK")
        if hook_env:
            args.hook = hook_env.split(":")

    # Validate openai_compatible provider requirements
    if args.provider == "openai_compatible":
        if args.model is None:
            parser.error("--model is required for openai_compatible provider")
        if args.host is None:
            parser.error("--host is required for openai_compatible provider")

    # Apply default host for ollama if not specified
    if args.provider == "ollama" and args.host is None:
        args.host = DEFAULT_OLLAMA_HOST

    # Determine model based on provider if not specified
    if args.model is None:
        args.model = {
            "bedrock": DEFAULT_BEDROCK_MODEL,
            "google": DEFAULT_GOOGLE_MODEL,
            "ollama": DEFAULT_OLLAMA_MODEL,
            "vertex": DEFAULT_VERTEX_MODEL,
        }[args.provider]

    # Load hooks if specified
    hooks = []
    if args.hook:
        for hook_path in args.hook:
            hooks.append(load_hook(hook_path))

    asyncio.run(run_agent(
        provider=args.provider,
        model=args.model,
        region=args.region,
        host=args.host,
        hooks=hooks,
        max_turns=args.max_turns,
        stats_file=args.stats_file,
    ))


if __name__ == "__main__":
    main()
