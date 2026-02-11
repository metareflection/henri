"""Google Cloud Vertex AI provider for Claude models."""

import os

from anthropic import AnthropicVertex

from henri.config import DEFAULT_VERTEX_MODEL, DEFAULT_VERTEX_REGION
from henri.providers.anthropic import AnthropicProvider


class VertexProvider(AnthropicProvider):
    """Vertex AI provider for Claude models — same as Anthropic but via Vertex."""

    name = "vertex"

    def __init__(
        self,
        model_id: str = DEFAULT_VERTEX_MODEL,
        region: str = DEFAULT_VERTEX_REGION,
        project: str | None = None,
        **kwargs,
    ):
        project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError(
                "Vertex provider requires GOOGLE_CLOUD_PROJECT env var or project parameter"
            )
        self.model_id = model_id
        self.client = AnthropicVertex(region=region, project_id=project)
