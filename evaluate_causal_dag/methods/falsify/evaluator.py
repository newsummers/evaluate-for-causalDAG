"""
Main causal-graph evaluator based on the Local Markov Condition.

Overview
--------
The :class:`FalsifyGraphEvaluator` accepts a directed acyclic graph and an
observed dataset.  It performs two kinds of assessments:

1. **Markov-violation check** — for every Local-Markov triplet
   (X, Z, Pa(X)) implied by the graph it runs a conditional independence (CI)
   test.  A *violation* occurs when the data reject the null hypothesis of
   independence, suggesting the graph is mis-specified.

2. **Edge suggestions** (optional) — for every directed edge *X → Y* it tests
   whether *X ⊥ Y | Pa(Y) \\ {X}*.  If the test cannot reject independence,
   the edge may be superfluous.

Caching
-------
An instance of :class:`~evaluate_causal_dag.methods.falsify.cache.CITestCache`
is associated with each evaluator.  The cache stores results keyed by the
symmetric triple (X, Y, conditioning_set).  When the same graph is evaluated
again — or a slightly modified graph is evaluated — already-computed results
are reused and only *new* triplets are tested.  Pass ``cache=None`` to disable
caching or share a cache instance across multiple evaluator instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx
import pandas as pd

from evaluate_causal_dag.methods.falsify.cache import CITestCache
from evaluate_causal_dag.methods.falsify.independence_tests import (
    CITestResult,
    IndependenceTestMethod,
    run_ci_test,
)
from evaluate_causal_dag.methods.falsify.triplet_selector import (
    EdgeSuggestionTriplet,
    LocalMarkovTriplet,
    get_edge_suggestion_triplets,
    get_local_markov_triplets,
)


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------


@dataclass
class EdgeSuggestion:
    """Suggestion to review a specific directed edge.

    Attributes
    ----------
    source, target:
        The edge being reviewed.
    ci_result:
        The CI test result that triggered this suggestion.
    message:
        Human-readable explanation.
    """

    source: str
    target: str
    ci_result: CITestResult
    message: str = "Consider removing or reversing this edge."

    def __repr__(self) -> str:
        return (
            f"EdgeSuggestion({self.source} → {self.target}: {self.message} "
            f"[{self.ci_result}])"
        )


@dataclass
class EvaluationResult:
    """Result of a single graph evaluation.

    Attributes
    ----------
    n_tests:
        Total number of CI tests that were run (cached results *not* counted).
    n_cached:
        Number of CI tests whose result was retrieved from the cache.
    test_results:
        All CI test results (both freshly computed and cached).
    violations:
        Subset of *test_results* where independence was rejected — i.e. where
        the graph's implied independence statement is contradicted by the data.
    suggestions:
        Edge suggestions (only populated when ``include_suggestions=True``).
    """

    n_tests: int
    n_cached: int
    test_results: list[CITestResult]
    violations: list[CITestResult]
    suggestions: list[EdgeSuggestion]

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def n_total(self) -> int:
        """Total triplets evaluated (tests + cache hits)."""
        return self.n_tests + self.n_cached

    @property
    def violation_rate(self) -> float:
        """Fraction of triplets that violated the Markov condition."""
        if self.n_total == 0:
            return 0.0
        return len(self.violations) / self.n_total

    @property
    def is_falsified(self) -> bool:
        """``True`` when at least one Markov violation was found."""
        return len(self.violations) > 0

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise human-readable evaluation summary."""
        lines = [
            "=== Causal Graph Evaluation Summary ===",
            f"  Total triplets evaluated : {self.n_total}",
            f"  Freshly computed         : {self.n_tests}",
            f"  Retrieved from cache     : {self.n_cached}",
            f"  Violations               : {len(self.violations)} "
            f"({self.violation_rate:.1%})",
            f"  Verdict                  : "
            f"{'FALSIFIED' if self.is_falsified else 'NOT FALSIFIED'}",
        ]
        if self.violations:
            lines.append("  Violations detail:")
            for v in self.violations:
                lines.append(f"    - {v}")
        if self.suggestions:
            lines.append(f"  Edge suggestions ({len(self.suggestions)}):")
            for s in self.suggestions:
                lines.append(f"    - {s}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(violations={len(self.violations)}/{self.n_total}, "
            f"falsified={self.is_falsified})"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class FalsifyGraphEvaluator:
    """Evaluate a causal DAG against observed data.

    The evaluator checks whether the conditional independence statements
    implied by the graph's Local Markov Condition are consistent with the
    observed data.  A built-in cache avoids redundant CI tests across
    successive evaluations of the same or similar graphs.

    Parameters
    ----------
    method:
        CI test method to use.  Defaults to Fisher's Z.
    alpha:
        Significance level for hypothesis tests.
    cache:
        A :class:`~evaluate_causal_dag.methods.falsify.cache.CITestCache`
        instance to use.  Pass ``None`` to start with a fresh private cache.
        Pass an existing instance to share the cache across evaluators.

    Examples
    --------
    >>> import networkx as nx, pandas as pd, numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> y = 0.8 * x + rng.normal(size=n)
    >>> z = 0.6 * y + rng.normal(size=n)
    >>> data = pd.DataFrame({"X": x, "Y": y, "Z": z})
    >>> dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
    >>> evaluator = FalsifyGraphEvaluator()
    >>> result = evaluator.evaluate(dag, data)
    >>> result.is_falsified
    False
    """

    def __init__(
        self,
        method: IndependenceTestMethod = IndependenceTestMethod.FISHER_Z,
        alpha: float = 0.05,
        cache: Optional[CITestCache] = None,
    ) -> None:
        self.method = method
        self.alpha = alpha
        self.cache: CITestCache = cache if cache is not None else CITestCache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dag: nx.DiGraph,
        data: pd.DataFrame,
        include_suggestions: bool = False,
    ) -> EvaluationResult:
        """Evaluate *dag* against *data*.

        Parameters
        ----------
        dag:
            The causal DAG to evaluate.  Node names must correspond to
            column names in *data*.
        data:
            Observed data.
        include_suggestions:
            When ``True`` also test each directed edge for necessity and
            return :class:`EdgeSuggestion` objects for potentially redundant
            edges.

        Returns
        -------
        EvaluationResult
        """
        triplets = get_local_markov_triplets(dag)
        all_results, n_tests, n_cached = self._run_lm_tests(data, triplets)

        violations = [r for r in all_results if not r.is_independent]

        suggestions: list[EdgeSuggestion] = []
        if include_suggestions:
            edge_triplets = get_edge_suggestion_triplets(dag)
            suggestions, s_new, s_cached = self._run_edge_suggestions(
                data, edge_triplets
            )
            n_tests += s_new
            n_cached += s_cached

        return EvaluationResult(
            n_tests=n_tests,
            n_cached=n_cached,
            test_results=all_results,
            violations=violations,
            suggestions=suggestions,
        )

    def clear_cache(self) -> None:
        """Clear all cached CI-test results."""
        self.cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_compute(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        conditioning_set,
    ) -> tuple[CITestResult, bool]:
        """Return (result, was_cached)."""
        cached = self.cache.get(x, y, conditioning_set)
        if cached is not None:
            return cached, True

        result = run_ci_test(
            data,
            x,
            y,
            conditioning_set,
            method=self.method,
            alpha=self.alpha,
        )
        self.cache.put(result)
        return result, False

    def _run_lm_tests(
        self,
        data: pd.DataFrame,
        triplets: list[LocalMarkovTriplet],
    ) -> tuple[list[CITestResult], int, int]:
        """Run CI tests for Local-Markov triplets.

        Returns (results, n_fresh, n_cached).
        """
        results: list[CITestResult] = []
        n_tests = 0
        n_cached = 0

        for triplet in triplets:
            result, from_cache = self._get_or_compute(
                data,
                triplet.node,
                triplet.other,
                triplet.parents,
            )
            results.append(result)
            if from_cache:
                n_cached += 1
            else:
                n_tests += 1

        return results, n_tests, n_cached

    def _run_edge_suggestions(
        self,
        data: pd.DataFrame,
        triplets: list[EdgeSuggestionTriplet],
    ) -> tuple[list[EdgeSuggestion], int, int]:
        """Run CI tests for edge-suggestion triplets.

        Returns (suggestions, n_fresh, n_cached).
        """
        suggestions: list[EdgeSuggestion] = []
        n_tests = 0
        n_cached = 0

        for triplet in triplets:
            result, from_cache = self._get_or_compute(
                data,
                triplet.source,
                triplet.target,
                triplet.other_parents,
            )
            if from_cache:
                n_cached += 1
            else:
                n_tests += 1

            if result.is_independent:
                suggestions.append(
                    EdgeSuggestion(
                        source=triplet.source,
                        target=triplet.target,
                        ci_result=result,
                        message=(
                            f"{triplet.source} ⊥ {triplet.target} | "
                            f"{{{', '.join(sorted(triplet.other_parents))}}} — "
                            "consider removing or reversing this edge."
                        ),
                    )
                )

        return suggestions, n_tests, n_cached
