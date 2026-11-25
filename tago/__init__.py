"""TAGO v3.8 - T4 Agent Graph Orchestrator evaluation lab.

Importing the package is the composition root: it registers the sandbox tool host and the
default tool registry into the ports-level registries used by the application ring.
"""
__version__ = "3.8.0"
import tago.infrastructure.tools  # noqa: E402,F401  (registers HOST_FACTORIES / TOOL_REGISTRIES)
