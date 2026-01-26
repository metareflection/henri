# Building Henri: A Step-by-Step Tutorial

This tutorial walks through how Henri works internally. By the end, you'll understand the core architecture of an AI coding assistant like Claude Code.

## The Big Picture

Henri is an **agentic loop**: it sends messages to an LLM, the LLM can request tool executions, Henri runs those tools, sends results back, and repeats until the LLM is done.

```
┌─────────────────────────────────────────────────────────┐
│                      User Input                         │
└─────────────────────────┬───────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    Agent Loop                           │
│  ┌───────────────────────────────────────────────────┐  │
│  │  Send messages + tools to LLM                     │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LLM responds with text and/or tool calls         │  │
│  └───────────────────────┬───────────────────────────┘  │
│                          ▼                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  If tool calls:                                   │  │
│  │    - Check permissions                            │  │
│  │    - Execute tools                                │  │
│  │    - Append results to messages                   │  │
│  │    - Loop back to send to LLM                     │  │
│  │  Else:                                            │  │
│  │    - Display response, done                       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Part 1: Message Types (`messages.py`)

Everything starts with defining how we represent conversations.

```python
@dataclass
class ToolCall:
    """A request from the LLM to execute a tool."""
    id: str        # Unique ID to match results back
    name: str      # Tool name, e.g., "bash"
    args: dict     # Arguments, e.g., {"command": "ls -la"}

@dataclass
class ToolResult:
    """The result of executing a tool."""
    tool_call_id: str  # Matches the ToolCall.id
    content: str       # Output from the tool
    is_error: bool = False

@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
```

**Key insight**: A message can contain text, tool calls (from the assistant), or tool results (sent back as a user message). This models the full conversation including tool interactions.

## Part 2: The Provider Abstraction (`providers/base.py`)

The provider abstracts LLM communication. Henri supports multiple providers (Bedrock, Google, Vertex, Ollama) through a common interface.

### The Provider Protocol

All providers implement the same abstract interface:

```python
@dataclass
class StreamEvent:
    """An event from a streaming response."""
    text: str = ""
    tool_calls: list[ToolCall] | None = None
    stop_reason: str | None = None


class Provider(ABC):
    """Abstract base class for LLM providers."""

    name: str  # Provider identifier

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        """Stream a response from the LLM."""
        pass
```

**Key insight**: This abstraction means the agent doesn't care which LLM it's talking to. Swap providers with a single flag: `--provider ollama`.

### Provider Registry

Providers are registered in `providers/__init__.py`:

```python
PROVIDERS: dict[str, type[Provider]] = {
    "bedrock": BedrockProvider,
    "google": GoogleProvider,
    "vertex": VertexProvider,
    "ollama": OllamaProvider,
}

def create_provider(name: str, **kwargs) -> Provider:
    return PROVIDERS[name](**kwargs)
```

### Example: Bedrock Provider

Each provider converts messages to its API's format:

```python
class BedrockProvider(Provider):
    name = "bedrock"

    def _message_to_bedrock(self, msg: Message) -> dict:
        content = []
        if msg.content:
            content.append({"text": msg.content})
        for tc in msg.tool_calls:
            content.append({
                "toolUse": {
                    "toolUseId": tc.id,
                    "name": tc.name,
                    "input": tc.args,
                }
            })
        # ... tool results similarly
        return {"role": msg.role, "content": content}
```

### Streaming Responses

Each provider streams events as they arrive:

```python
async def stream(self, messages, tools, system=""):
    response = self.client.converse_stream(...)

    for event in response["stream"]:
        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                yield StreamEvent(text=delta["text"])
        # ... handle tool calls and stop events
```

**Key insight**: Streaming lets you show text as it arrives, giving a responsive feel. Tool calls come as structured events at the end.

## Part 3: Tools (`tools/base.py`)

Tools are how the LLM interacts with the world.

### Tool Definition

Each tool declares its interface via JSON Schema:

```python
class Tool(ABC):
    name: str
    description: str
    parameters: dict  # JSON Schema
    requires_permission: bool = False

    @abstractmethod
    def execute(self, **kwargs) -> str:
        pass
```

### Example: Bash Tool

```python
class BashTool(Tool):
    name = "bash"
    description = "Execute a shell command and return its output."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
        },
        "required": ["command"],
    }
    requires_permission = True  # Dangerous! Ask user first.

    def execute(self, command: str) -> str:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.stdout + result.stderr
```

**Key insight**: The LLM sees the `description` and `parameters` schema. It decides when to call tools and with what arguments. Your `execute` method just runs the logic.

### Converting Tools for Bedrock

```python
def _tools_to_bedrock(self, tools: list[Tool]) -> list[dict]:
    return [
        {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.parameters},
            }
        }
        for tool in tools
    ]
```

## Part 4: Permissions (`permissions.py`)

Before running dangerous tools, ask the user:

```python
@dataclass
class PermissionManager:
    allowed_tools: set[str] = field(default_factory=set)
    allow_all: bool = False

    def check(self, tool: Tool, call: ToolCall) -> bool:
        if not tool.requires_permission:
            return True
        if self.allow_all or tool.name in self.allowed_tools:
            return True
        return self._prompt_user(tool, call)

    def _prompt_user(self, tool: Tool, call: ToolCall) -> bool:
        # Show tool name and arguments
        # Ask: (y)es / (n)o / (a)lways / (A)ll
        response = input("Allow? ")
        if response == "a":
            self.allowed_tools.add(tool.name)
        elif response == "A":
            self.allow_all = True
        return response in ("y", "a", "A")
```

**Key insight**: This is a simple session-based permission model. Claude Code has more sophisticated file pattern matching and persistent permissions.

## Part 5: The Agent Loop (`agent.py`)

This ties everything together:

```python
async def chat(self, user_input: str) -> None:
    self.messages.append(Message.user(user_input))

    while True:
        # 1. Send to LLM and stream response
        response_text = ""
        tool_calls = []

        async for event in self.provider.stream(
            self.messages,
            self.tools,
            system=SYSTEM_PROMPT,
        ):
            if event.text:
                print(event.text, end="")  # Stream to terminal
                response_text += event.text
            if event.tool_calls:
                tool_calls = event.tool_calls

        # 2. Record assistant's response
        self.messages.append(Message.assistant(response_text, tool_calls))

        # 3. If no tool calls, we're done
        if not tool_calls:
            break

        # 4. Execute tool calls
        results = []
        for call in tool_calls:
            tool = self.tools_by_name[call.name]

            # Check permission
            if not self.permissions.check(tool, call):
                results.append(ToolResult(
                    tool_call_id=call.id,
                    content="[permission denied]",
                    is_error=True,
                ))
                continue

            # Execute
            result = tool.execute(**call.args)
            results.append(ToolResult(tool_call_id=call.id, content=result))

        # 5. Add results and loop back
        self.messages.append(Message.tool_result(results))
```

### How the Loop Terminates

The loop runs until the LLM responds without any tool calls. Each iteration:

1. Send full conversation history to LLM
2. LLM streams back text and/or tool calls
3. If tool calls: execute them, append results, loop again
4. If no tool calls: break out, we're done

**Key insight**: The entire agent is just a `while True` loop. Stream a response, execute any tool calls, repeat. That's it.

### Note: `stop_reason`

`StreamEvent` also has a `stop_reason` field (`end_turn`, `tool_use`, or `max_tokens`). Henri doesn't use it yet, but could in the future (e.g., to warn users when a response is truncated).

## Part 6: Adding a New Tool

Want to add a new tool? Here's the pattern:

```python
class GrepTool(Tool):
    name = "grep"
    description = "Search for a pattern in files."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
            "path": {
                "type": "string",
                "description": "File or directory to search",
            },
        },
        "required": ["pattern", "path"],
    }
    requires_permission = False  # Reading is safe

    def execute(self, pattern: str, path: str) -> str:
        result = subprocess.run(
            ["grep", "-r", pattern, path],
            capture_output=True,
            text=True,
        )
        return result.stdout or "(no matches)"
```

Then add it to `get_default_tools()`:

```python
def get_default_tools() -> list[Tool]:
    return [BashTool(), ReadFileTool(), WriteFileTool(), GrepTool()]
```

## Part 7: Adding a New Provider

Henri already has three providers. Here's how to add another:

### 1. Create the Provider Class

```python
# providers/my_provider.py
from henri.providers.base import Provider, StreamEvent

class MyProvider(Provider):
    name = "my_provider"

    def __init__(self, model_id: str, api_key: str | None = None):
        self.model_id = model_id
        self.client = SomeClient(api_key=api_key)

    def _messages_to_api(self, messages: list[Message]) -> list[dict]:
        """Convert Henri messages to API format."""
        # Each API has its own format - this is the main work
        return [...]

    async def stream(
        self,
        messages: list[Message],
        tools: list[Tool],
        system: str = "",
    ) -> AsyncIterator[StreamEvent]:
        response = await self.client.chat(
            messages=self._messages_to_api(messages),
            tools=self._tools_to_api(tools),
        )
        async for chunk in response:
            if chunk.text:
                yield StreamEvent(text=chunk.text)
            if chunk.tool_calls:
                yield StreamEvent(tool_calls=chunk.tool_calls)
        yield StreamEvent(stop_reason="end_turn")
```

### 2. Register in the Provider Registry

```python
# providers/__init__.py
from .my_provider import MyProvider

PROVIDERS["my_provider"] = MyProvider
```

### 3. Add Config Defaults

```python
# config.py
DEFAULT_MY_PROVIDER_MODEL = "some-model"
```

The key insight: each provider's job is to translate between Henri's message format and the LLM API's format. The agent loop doesn't change.

## Part 8: Hooks

Hooks let you extend Henri without modifying core code. They're Python files that define special variables to customize tools and permissions.

### Using Hooks

```bash
henri --hook hooks/dafny.py
henri --hook hooks/dafny.py hooks/bench.py  # Multiple hooks
```

### Hook Variables

Hooks can define any of these variables:

| Variable | Type | Effect |
|----------|------|--------|
| `TOOLS` | `list[Tool]` | Tools to add |
| `REMOVE_TOOLS` | `set[str]` | Tool names to remove |
| `PATH_BASED` | `set[str]` | Tools that get per-path "always allow" |
| `AUTO_ALLOW_CWD` | `set[str]` | Tools to auto-allow within cwd |
| `AUTO_ALLOW` | `set[str]` | Tools to always allow (no prompts) |
| `REJECT_PROMPTS` | `bool` | Reject (don't prompt) for permissions |
| `SYSTEM_PROMPT` | `str` | Extra text appended to system prompt |

### Example: Adding a Domain-Specific Tool

Here's `hooks/dafny.py` - adds a Dafny verification tool:

```python
"""Usage: henri --hook hooks/dafny.py"""

import subprocess
from henri.tools.base import Tool


class DafnyVerifyTool(Tool):
    name = "dafny_verify"
    description = "Run 'dafny verify' on a Dafny file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the .dfy file to verify",
            },
        },
        "required": ["path"],
    }
    requires_permission = True

    def execute(self, path: str) -> str:
        try:
            result = subprocess.run(
                ["dafny", "verify", path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            return result.stdout + result.stderr or "(no output)"
        except FileNotFoundError:
            return "[error: dafny not found]"


# Add the tool
TOOLS = [DafnyVerifyTool()]

# Per-path permissions (not global "always allow")
PATH_BASED = {"dafny_verify"}

# Auto-allow within current working directory
AUTO_ALLOW_CWD = {"dafny_verify"}
```

### Example: Restricting for Benchmarks

Here's `hooks/bench.py` - for non-interactive/automated use:

```python
"""Usage: echo "task" | henri --hook hooks/bench.py"""

# Remove dangerous tools
REMOVE_TOOLS = {"bash", "web_fetch"}

# Auto-allow file writes within cwd
AUTO_ALLOW_CWD = {"write_file", "edit_file"}

# Don't prompt - reject if permission needed
REJECT_PROMPTS = True
```

### How Hooks Are Loaded

The CLI loads hooks as Python modules and merges their settings:

```python
def load_hook(hook_path: str):
    spec = importlib.util.spec_from_file_location("hook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# In run_agent():
for hook in hooks:
    if hasattr(hook, "TOOLS"):
        tools = tools + hook.TOOLS
    if hasattr(hook, "REMOVE_TOOLS"):
        tools = [t for t in tools if t.name not in hook.REMOVE_TOOLS]
    # ... similarly for permission settings
```

**Key insight**: Hooks compose. You can use multiple hooks together - their `TOOLS` lists concatenate and their permission sets merge with `|=`.

## Exercises

- **Add conversation history**: Save/load `self.messages` to JSON to resume conversations.

- **Add a new provider**: Try adding OpenAI, Anthropic direct API, or another LLM service.

- **Add file pattern permissions**: Instead of per-tool, allow "write to *.py files" patterns.

- **Add streaming tool output**: For long-running bash commands, stream output as it happens.

- **Compare models**: Run the same task across different providers/models and compare results. Which is faster? More accurate? More verbose?

- **Add a TODO system**: Inject a `<todo-list>` into the system prompt that the agent maintains. This helps prevent the agent from stopping prematurely or forgetting subtasks. See how it affects task completion rates.

- **Add context management**: When conversations get long, the context window fills up. Implement summarization of old messages, or a sliding window that keeps recent messages plus a summary of earlier context.

- **Add retry with backoff**: When a tool fails (e.g., network error, rate limit), retry with exponential backoff instead of immediately returning the error to the LLM.

- **Add self-verification**: After the agent says it's done, automatically run verification (tests, linter, type checker) and feed failures back to continue the loop. Compare completion rates with and without this.

