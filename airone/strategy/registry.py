"""
AirOne Strategy Registry
Central catalogue of all available compression strategies.
"""

from __future__ import annotations

from airone.compressors.base import BaseCompressor
from airone.exceptions import StrategyError


class StrategyRegistry:
    """
    Maintains a mapping of strategy_name → compressor instance.
    New strategies (procedural, neural, etc.) are registered here.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, BaseCompressor] = {}

    def register(self, compressor: BaseCompressor) -> None:
        self._strategies[compressor.name] = compressor

    def get(self, name: str) -> BaseCompressor:
        if name not in self._strategies:
            raise StrategyError(f"Strategy '{name}' is not registered.")
        return self._strategies[name]

    def list_names(self) -> list[str]:
        return list(self._strategies.keys())

    def list_all(self) -> list[BaseCompressor]:
        return list(self._strategies.values())
