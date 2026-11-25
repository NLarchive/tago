"""Ring 1 - ports: the two interfaces the application layer depends on.

The orchestrator never imports a concrete backend or tool host. Anything that satisfies
these protocols (Ollama, an OpenAI-compatible API, Anthropic, a scripted test double,
or a future MCP tool host) can be injected.
"""

# %% cell: ports
from typing import Protocol, runtime_checkable

@runtime_checkable
class ChatBackend(Protocol):
    """Structured-output chat port. Must return an instance of `schema` or raise."""
    model: str
    records: list
    def chat(self, role, system, payload, schema, seed=0, temperature=0): ...

@runtime_checkable
class ToolHost(Protocol):
    """Per-case tool host port. `call` returns an observation dict with a stable id."""
    pack: object
    observations: list
    registry: object
    def call(self, request): ...

# Composition registries: infrastructure registers concrete implementations here at import time,
# so the application ring can create a host or find the tool registry without importing infrastructure.
HOST_FACTORIES = {}
TOOL_REGISTRIES = {}
