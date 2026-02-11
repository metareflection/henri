# Henri

Henri is a small, hackable agent CLI in Python, with explicit control via tools, permissions, and hooks.
It comes with [a tutorial](TUTORIAL.md)
and is inspired by Claude Code.

## Features

- **Multiple LLM providers** - Anthropic API, AWS Bedrock, Google Gemini, Vertex AI, Ollama, OpenAI-compatible (VLLM, etc.)
- **Streaming responses** - Real-time token streaming
- **Tool system** - bash, file read/write/edit capabilities, grep/glob, ...
- **Permission management** - Prompt or auto-deny operations
- **Hook system** - Add custom tools and configure permissions via `--hook`
- **Clean architecture** - Easy to understand and extend

## Installation

```bash
pip install -e .
brew install ripgrep  # for the grep tool
```

## Usage

```bash
# Anthropic API
henri --provider anthropic

# AWS Bedrock (default)
henri

# Google Gemini
henri --provider google

# Vertex AI
henri --provider vertex

# Ollama
henri --provider ollama

# OpenAI-compatible server (VLLM, etc.)
henri --provider openai_compatible --model <model-name> --host <server-url>
```

### Provider Setup

**Anthropic API**:
- Set `ANTHROPIC_API_KEY`

**AWS Bedrock** (default):
- Configure AWS credentials (`aws configure` or environment variables)
- Ensure access to Claude models in your region

**Google Gemini**:
- Set `GOOGLE_API_KEY` for Google AI API, or
- Set `GOOGLE_CLOUD_PROJECT` for Vertex AI

**Vertex AI**:
- Set `GOOGLE_CLOUD_PROJECT`

**Ollama**:
- Install and run [Ollama](https://ollama.ai) locally
- Pull a model: `ollama pull qwen3-coder:30b`

**OpenAI-compatible server**
- Start an OpenAI-compatible server with tool calling, e.g.,: `vllm serve $MODEL_PATH --dtype auto --max_model_len 4096 --served-model-name $MODEL_NAME --tool-call-parser $MODEL_TOOL_CALL_PARSER --enable-auto-tool-choice`


### Example Session

```
> What files are in the current directory?

▶ bash(command='ls -la')
┌──────────────────────────────────────┐
│ total 16                             │
│ drwxr-xr-x  5 user staff  160 Jan  3 │
│ -rw-r--r--  1 user staff  123 Jan  3 │
│ ...                                  │
└──────────────────────────────────────┘

There are 5 files in the current directory...
```

### Permission System

When Henri wants to execute a tool that requires permission (like `bash` or `write_file`), you'll be prompted:

- `y` - Allow this execution
- `n` - Deny this execution
- `a` - Always allow (scope depends on tool: exact command for bash, per-path for file tools)
- `A` - Allow all tools for the session

## Architecture

```
henri/
├── messages.py      # Core data types (Message, ToolCall, ToolResult)
├── providers/
│   ├── base.py       # Provider abstract base class
│   ├── anthropic.py  # Anthropic API (+ base for Vertex)
│   ├── bedrock.py    # AWS Bedrock
│   ├── google.py     # Google Gemini
│   ├── vertex.py     # Vertex AI (extends anthropic.py)
│   └── ollama.py     # Ollama
├── tools/
│   └── base.py      # Tool base class + built-in tools
├── permissions.py   # Permission management
├── agent.py         # Main conversation loop
└── cli.py           # Entry point
```

### Adding New Tools

Create a new tool by subclassing `Tool`:

```python
from henri.tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "arg1": {"type": "string", "description": "First argument"},
        },
        "required": ["arg1"],
    }
    requires_permission = True  # Set to False for safe operations

    def execute(self, arg1: str) -> str:
        # Your implementation here
        return f"Result: {arg1}"
```

Then add it to the tools list in `agent.py` or pass it when creating the Agent.

### Adding New Providers

Subclass `Provider` and implement the `stream()` method:

```python
from henri.providers.base import Provider, StreamEvent

class MyProvider(Provider):
    name = "my_provider"

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        # Your implementation
        yield StreamEvent(text="Hello")
        yield StreamEvent(stop_reason="end_turn")
```

Then register it in `providers/__init__.py`.

## Configuration

```bash
# Provider selection
henri --provider anthropic|bedrock|google|vertex|ollama

# Model override
henri --model <model-id>

# Provider-specific options
henri --region us-east-1             # AWS Bedrock
henri --region us-east5              # Vertex AI
henri --host http://localhost:11434  # Ollama

# Limit turns (for benchmarking)
henri --max-turns 10                 # Stop after 10 turns (default: unlimited)

# Load hooks (can be used multiple times)
henri --hook hooks/dafny.py          # Add dafny_verify tool
henri --hook hooks/dafny.py --hook hooks/bench.py  # Combine hooks
```

On exit, Henri prints metrics: `Turns: X | Tokens: Y in, Z out`

### Hooks

Hooks are Python files that customize Henri without modifying core code. They can:

- Add custom tools (`TOOLS = [MyTool()]`)
- Remove tools (`REMOVE_TOOLS = {"bash"}`)
- Configure auto-allow permissions (`AUTO_ALLOW_CWD = {"my_tool"}`)
- Reject permission prompts for automation (`REJECT_PROMPTS = True`)
- Add to the system prompt (`SYSTEM_PROMPT = "extra instructions..."`)

See `hooks/` for examples and the [tutorial](TUTORIAL.md#part-8-hooks) for details.

### Used in

- [Dafny Sketcher](https://github.com/namin/dafny-sketcher)
  - [henri.py](https://github.com/namin/dafny-sketcher/blob/main/vfp/henri.py) - Hook adding `dafny_sketcher` tool
  - [henri_bench.py](https://github.com/namin/dafny-sketcher/blob/main/vfp/henri_bench.py) - Benchmark harness for evaluating proof synthesis

### Links

- [Supported AWS Bedrock models](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html)
- [Google Cloud Model Garden](https://console.cloud.google.com/vertex-ai/model-garden)
- [Ollama model library](https://ollama.ai/library)

## License

MIT
