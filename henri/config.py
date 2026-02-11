"""Configuration defaults for Henri."""

import os
from dataclasses import dataclass


def env_var(name: str, default=None):
    """Get environment variable with HENRI_ prefix."""
    return os.environ.get(f"HENRI_{name}", default)


@dataclass
class ProviderConfig:
    """Resolved provider configuration."""

    provider: str
    model: str
    region: str | None = None
    host: str | None = None
    max_turns: int | None = None


# Default provider
DEFAULT_PROVIDER = "bedrock"

# AWS Bedrock defaults
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_BEDROCK_REGION = "us-east-1"

# Google Cloud defaults
DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash"
DEFAULT_GOOGLE_LOCATION = "us-central1"

# Ollama defaults
DEFAULT_OLLAMA_MODEL = "qwen3-coder:30b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Vertex AI defaults (for Claude models)
DEFAULT_VERTEX_MODEL = "claude-sonnet-4-5"
DEFAULT_VERTEX_REGION = "us-east5"

# Direct Anthropic API defaults
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"


def get_provider_config(
    provider: str | None = None,
    model: str | None = None,
    region: str | None = None,
    host: str | None = None,
    max_turns: int | None = None,
) -> ProviderConfig:
    """Get resolved provider configuration from args and environment variables.

    Arguments take precedence over HENRI_* environment variables, which take
    precedence over built-in defaults.

    Args:
        provider: LLM provider name (or HENRI_PROVIDER env var)
        model: Model ID (or HENRI_MODEL env var)
        region: Region for cloud providers (or HENRI_REGION env var)
        host: Host URL for Ollama/OpenAI-compatible (or HENRI_HOST env var)
        max_turns: Maximum conversation turns (or HENRI_MAX_TURNS env var)

    Returns:
        ProviderConfig with all values resolved
    """
    # Resolve provider (always has a default)
    resolved_provider: str = provider or env_var("PROVIDER") or DEFAULT_PROVIDER

    # Resolve region
    resolved_region = region or env_var("REGION")

    # Resolve host (with Ollama default)
    resolved_host = host or env_var("HOST")
    if resolved_provider == "ollama" and resolved_host is None:
        resolved_host = DEFAULT_OLLAMA_HOST

    # Resolve model (with provider-specific defaults)
    resolved_model: str | None = model or env_var("MODEL")
    if resolved_model is None:
        model_defaults: dict[str, str] = {
            "bedrock": DEFAULT_BEDROCK_MODEL,
            "google": DEFAULT_GOOGLE_MODEL,
            "ollama": DEFAULT_OLLAMA_MODEL,
            "vertex": DEFAULT_VERTEX_MODEL,
            "anthropic": DEFAULT_ANTHROPIC_MODEL,
        }
        resolved_model = model_defaults.get(resolved_provider)

    # Resolve max_turns
    resolved_max_turns = max_turns
    if resolved_max_turns is None:
        max_turns_env = env_var("MAX_TURNS")
        if max_turns_env:
            resolved_max_turns = int(max_turns_env)

    return ProviderConfig(
        provider=resolved_provider,
        model=resolved_model,  # type: ignore[arg-type]  # None valid for openai_compatible validation
        region=resolved_region,
        host=resolved_host,
        max_turns=resolved_max_turns,
    )
