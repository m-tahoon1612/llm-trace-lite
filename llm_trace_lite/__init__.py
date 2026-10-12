"""Record OpenAI-compatible chat completion calls into SQLite."""

from .client import TracedClient
from .store import TraceStore

__all__ = ["TracedClient", "TraceStore"]
__version__ = "0.1.0"
