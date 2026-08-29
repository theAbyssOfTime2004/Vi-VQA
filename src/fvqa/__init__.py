"""FVQA: Fact-based Visual Question Answering with knowledge-graph traversal, on Qwen3-VL."""

from fvqa.config import Config, ConfigError, load_config

__version__ = "0.2.0"
__all__ = ["Config", "ConfigError", "__version__", "load_config"]
