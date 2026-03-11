"""
Cache for conditional independence test results.

Stores computed CITestResult objects keyed by (X, Y, conditioning_set) so that
subsequent evaluations of graphs that share many of the same triplets can reuse
already-computed results instead of re-running expensive statistical tests.

The cache key is symmetric in X and Y, meaning cached(X, Y, S) is also returned
when querying (Y, X, S).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from evaluate_causal_dag.methods.falsify.independence_tests import CITestResult


class CITestCache:
    """
    Thread-unsafe, in-memory cache for :class:`CITestResult` objects.

    Key schema
    ----------
    The key is a 3-tuple ``(x_canonical, y_canonical, cond_set)`` where
    ``x_canonical <= y_canonical`` (lexicographic) and ``cond_set`` is a
    :class:`frozenset` of conditioning variable names.  This makes the key
    symmetric: ``cache(X, Y, S)`` and ``cache(Y, X, S)`` map to the same slot.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str, frozenset[str]], "CITestResult"] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(
        x: str, y: str, conditioning_set
    ) -> tuple[str, str, frozenset[str]]:
        a, b = (x, y) if x <= y else (y, x)
        return (a, b, frozenset(conditioning_set))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, x: str, y: str, conditioning_set
    ) -> Optional["CITestResult"]:
        """Return the cached result or ``None`` if not present."""
        return self._store.get(self._make_key(x, y, conditioning_set))

    def put(self, result: "CITestResult") -> None:
        """Insert *result* into the cache using its own (x, y, conditioning_set)."""
        key = self._make_key(result.x, result.y, result.conditioning_set)
        self._store[key] = result

    def has(self, x: str, y: str, conditioning_set) -> bool:
        """Return ``True`` if a result for this triplet is already cached."""
        return self._make_key(x, y, conditioning_set) in self._store

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:  # pragma: no cover
        return f"CITestCache(size={len(self)})"
