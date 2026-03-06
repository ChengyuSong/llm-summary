"""Thread pool for parallel LLM queries."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any


class LLMPool:
    """Thin wrapper around ThreadPoolExecutor for parallel LLM calls."""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)

    def __enter__(self) -> LLMPool:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown(wait=True)
