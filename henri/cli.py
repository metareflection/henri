"""CLI entry point for Henri."""

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

from henri.agent import run_agent
from henri.config import (
    DEFAULT_PROVIDER,
    env_var,
    get_provider_config,
)
from henri.providers import PROVIDERS


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

    # Apply env vars for hooks (list splitting)
    if args.hook is None:
        hook_env = env_var("HOOK")
        if hook_env:
            args.hook = hook_env.split(":")

    # Get resolved provider configuration
    config = get_provider_config(
        provider=args.provider,
        model=args.model,
        region=args.region,
        host=args.host,
        max_turns=args.max_turns,
    )

    # Validate openai_compatible provider requirements
    if config.provider == "openai_compatible":
        if config.model is None:
            parser.error("--model is required for openai_compatible provider")
        if config.host is None:
            parser.error("--host is required for openai_compatible provider")

    # Load hooks if specified
    hooks = []
    if args.hook:
        for hook_path in args.hook:
            hooks.append(load_hook(hook_path))

    asyncio.run(run_agent(
        provider=config.provider,
        model=config.model,
        region=config.region,
        host=config.host,
        hooks=hooks,
        max_turns=config.max_turns,
        stats_file=args.stats_file,
    ))


if __name__ == "__main__":
    main()
