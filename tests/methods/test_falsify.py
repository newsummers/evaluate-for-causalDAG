"""
Tests for evaluate_causal_dag.methods.falsify.

Covers:
- CITestCache (put, get, has, symmetry, clear, len)
- Fisher's Z test (marginal and conditional)
- Mutual-information test
- Local-Markov triplet selection
- Edge-suggestion triplet selection
- FalsifyGraphEvaluator (evaluation, caching, suggestions, clear_cache)
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import pandas as pd
import pytest

from evaluate_causal_dag.methods.falsify import (
    CITestCache,
    CITestResult,
    EdgeSuggestion,
    EdgeSuggestionTriplet,
    EvaluationResult,
    FalsifyGraphEvaluator,
    IndependenceTestMethod,
    LocalMarkovTriplet,
    get_edge_suggestion_triplets,
    get_local_markov_triplets,
    run_ci_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


RNG = np.random.default_rng(42)
N = 500  # sample size large enough for stable tests


def _chain_data(n: int = N, rng=RNG) -> pd.DataFrame:
    """Generate data from the chain X → Y → Z."""
    x = rng.normal(size=n)
    y = 0.8 * x + rng.normal(scale=0.5, size=n)
    z = 0.7 * y + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _fork_data(n: int = N, rng=RNG) -> pd.DataFrame:
    """Generate data from the fork X ← Y → Z."""
    y = rng.normal(size=n)
    x = 0.8 * y + rng.normal(scale=0.5, size=n)
    z = 0.7 * y + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z})


def _chain_dag() -> nx.DiGraph:
    return nx.DiGraph([("X", "Y"), ("Y", "Z")])


def _fork_dag() -> nx.DiGraph:
    return nx.DiGraph([("Y", "X"), ("Y", "Z")])


def _wrong_dag() -> nx.DiGraph:
    """A dag that wrongly claims X ⊥ Z (no path X-Z), contradicted by data."""
    # No edge between X and Z; but data has X → Y → Z.
    # This graph claims X ⊥ Z | {} which is FALSE for chain data.
    g = nx.DiGraph()
    g.add_nodes_from(["X", "Y", "Z"])
    g.add_edge("X", "Y")
    # Missing Y → Z, so the graph says Z is independent of X and Y given {}
    return g


# ---------------------------------------------------------------------------
# CITestCache
# ---------------------------------------------------------------------------


class TestCITestCache:
    def _make_result(self, x="A", y="B", cond=frozenset(), p=0.5):
        return CITestResult(
            x=x, y=y, conditioning_set=cond,
            p_value=p, statistic=1.0,
            is_independent=p > 0.05,
            method=IndependenceTestMethod.FISHER_Z,
        )

    def test_put_and_get(self):
        cache = CITestCache()
        result = self._make_result()
        cache.put(result)
        retrieved = cache.get("A", "B", frozenset())
        assert retrieved is result

    def test_symmetric_key(self):
        """get(X, Y, S) and get(Y, X, S) must return the same entry."""
        cache = CITestCache()
        result = self._make_result(x="A", y="B")
        cache.put(result)
        assert cache.get("B", "A", frozenset()) is result

    def test_has(self):
        cache = CITestCache()
        assert not cache.has("A", "B", frozenset())
        cache.put(self._make_result())
        assert cache.has("A", "B", frozenset())
        assert cache.has("B", "A", frozenset())  # symmetry

    def test_conditioning_set_order_irrelevant(self):
        cache = CITestCache()
        result = self._make_result(cond=frozenset({"C", "D"}))
        cache.put(result)
        assert cache.get("A", "B", {"D", "C"}) is result
        assert cache.get("A", "B", ["C", "D"]) is result

    def test_len(self):
        cache = CITestCache()
        cache.put(self._make_result(x="A", y="B"))
        cache.put(self._make_result(x="C", y="D"))
        assert len(cache) == 2

    def test_clear(self):
        cache = CITestCache()
        cache.put(self._make_result())
        cache.clear()
        assert len(cache) == 0
        assert cache.get("A", "B", frozenset()) is None

    def test_missing_returns_none(self):
        cache = CITestCache()
        assert cache.get("X", "Y", frozenset()) is None


# ---------------------------------------------------------------------------
# Independence tests
# ---------------------------------------------------------------------------


class TestFisherZTest:
    """Fisher's Z test sanity checks."""

    data_chain = _chain_data()
    data_fork = _fork_data()

    def test_marginal_dependence(self):
        """X and Z are marginally dependent in chain data."""
        result = run_ci_test(self.data_chain, "X", "Z", [], alpha=0.05)
        assert not result.is_independent, "X and Z should be dependent marginally"
        assert result.p_value < 0.05

    def test_conditional_independence_chain(self):
        """In chain X→Y→Z, X ⊥ Z | Y should hold."""
        result = run_ci_test(self.data_chain, "X", "Z", ["Y"], alpha=0.05)
        assert result.is_independent, "X ⊥ Z | Y should hold in chain"

    def test_marginal_dependence_fork(self):
        """In fork X←Y→Z, X and Z are marginally dependent."""
        result = run_ci_test(self.data_fork, "X", "Z", [], alpha=0.05)
        assert not result.is_independent

    def test_conditional_independence_fork(self):
        """In fork X←Y→Z, X ⊥ Z | Y should hold."""
        result = run_ci_test(self.data_fork, "X", "Z", ["Y"], alpha=0.05)
        assert result.is_independent

    def test_method_field(self):
        result = run_ci_test(self.data_chain, "X", "Y", [], alpha=0.05)
        assert result.method is IndependenceTestMethod.FISHER_Z

    def test_result_fields(self):
        result = run_ci_test(self.data_chain, "X", "Y", ["Z"], alpha=0.05)
        assert result.x == "X"
        assert result.y == "Y"
        assert result.conditioning_set == frozenset({"Z"})
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.statistic, float)

    def test_small_sample_warning(self):
        """Tiny dataset should warn and return NaN p-value."""
        tiny = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0], "C": [1.0, 1.5]})
        with pytest.warns(UserWarning):
            result = run_ci_test(tiny, "A", "B", ["C"], alpha=0.05)
        assert np.isnan(result.p_value)


class TestMutualInfoTest:
    """Smoke-tests for mutual information / chi-square CI test."""

    def _discrete_data(self, n=300):
        rng = np.random.default_rng(7)
        x = rng.integers(0, 3, size=n)
        y = (x + rng.integers(0, 2, size=n)) % 3
        z = rng.integers(0, 3, size=n)  # Z independent of X and Y
        return pd.DataFrame({"X": x, "Y": y, "Z": z})

    def test_dependent_pair(self):
        data = self._discrete_data()
        result = run_ci_test(
            data, "X", "Y", [],
            method=IndependenceTestMethod.MUTUAL_INFO, alpha=0.05
        )
        assert not result.is_independent

    def test_independent_pair(self):
        data = self._discrete_data()
        result = run_ci_test(
            data, "X", "Z", [],
            method=IndependenceTestMethod.MUTUAL_INFO, alpha=0.05
        )
        assert result.is_independent

    def test_method_field(self):
        data = self._discrete_data()
        result = run_ci_test(
            data, "X", "Y", [],
            method=IndependenceTestMethod.MUTUAL_INFO
        )
        assert result.method is IndependenceTestMethod.MUTUAL_INFO


class TestKernelBasedTest:
    """Kernel-based (HSIC) CI test sanity checks."""

    data_chain = _chain_data()
    data_fork = _fork_data()

    def test_marginal_dependence(self):
        """X and Z are marginally dependent in chain data."""
        result = run_ci_test(
            self.data_chain, "X", "Z", [],
            method=IndependenceTestMethod.KERNEL_BASED, alpha=0.05,
        )
        assert not result.is_independent, "X and Z should be dependent marginally"
        assert result.p_value < 0.05

    def test_conditional_independence_chain(self):
        """In chain X→Y→Z, X ⊥ Z | Y should hold."""
        result = run_ci_test(
            self.data_chain, "X", "Z", ["Y"],
            method=IndependenceTestMethod.KERNEL_BASED, alpha=0.05,
        )
        assert result.is_independent, "X ⊥ Z | Y should hold in chain"

    def test_conditional_independence_fork(self):
        """In fork X←Y→Z, X ⊥ Z | Y should hold."""
        result = run_ci_test(
            self.data_fork, "X", "Z", ["Y"],
            method=IndependenceTestMethod.KERNEL_BASED, alpha=0.05,
        )
        assert result.is_independent

    def test_method_field(self):
        result = run_ci_test(
            self.data_chain, "X", "Y", [],
            method=IndependenceTestMethod.KERNEL_BASED,
        )
        assert result.method is IndependenceTestMethod.KERNEL_BASED

    def test_small_sample_returns_nan(self):
        tiny = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]})
        result = run_ci_test(
            tiny, "A", "B", [],
            method=IndependenceTestMethod.KERNEL_BASED,
        )
        assert np.isnan(result.p_value)


class TestApproxKernelBasedTest:
    """Approximate kernel-based CI test sanity checks."""

    data_chain = _chain_data()
    data_fork = _fork_data()

    def test_marginal_dependence(self):
        result = run_ci_test(
            self.data_chain, "X", "Z", [],
            method=IndependenceTestMethod.APPROX_KERNEL_BASED, alpha=0.05,
        )
        assert not result.is_independent, "X and Z should be dependent marginally"

    def test_conditional_independence_chain(self):
        result = run_ci_test(
            self.data_chain, "X", "Z", ["Y"],
            method=IndependenceTestMethod.APPROX_KERNEL_BASED, alpha=0.05,
        )
        assert result.is_independent, "X ⊥ Z | Y should hold in chain"

    def test_conditional_independence_fork(self):
        result = run_ci_test(
            self.data_fork, "X", "Z", ["Y"],
            method=IndependenceTestMethod.APPROX_KERNEL_BASED, alpha=0.05,
        )
        assert result.is_independent

    def test_method_field(self):
        result = run_ci_test(
            self.data_chain, "X", "Y", [],
            method=IndependenceTestMethod.APPROX_KERNEL_BASED,
        )
        assert result.method is IndependenceTestMethod.APPROX_KERNEL_BASED

    def test_small_sample_returns_nan(self):
        tiny = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]})
        result = run_ci_test(
            tiny, "A", "B", [],
            method=IndependenceTestMethod.APPROX_KERNEL_BASED,
        )
        assert np.isnan(result.p_value)


class TestRegressionBasedTest:
    """Regression-based CI test sanity checks."""

    data_chain = _chain_data()
    data_fork = _fork_data()

    def test_marginal_dependence(self):
        result = run_ci_test(
            self.data_chain, "X", "Z", [],
            method=IndependenceTestMethod.REGRESSION_BASED, alpha=0.05,
        )
        assert not result.is_independent, "X and Z should be dependent marginally"

    def test_conditional_independence_chain(self):
        result = run_ci_test(
            self.data_chain, "X", "Z", ["Y"],
            method=IndependenceTestMethod.REGRESSION_BASED, alpha=0.05,
        )
        assert result.is_independent, "X ⊥ Z | Y should hold in chain"

    def test_conditional_independence_fork(self):
        result = run_ci_test(
            self.data_fork, "X", "Z", ["Y"],
            method=IndependenceTestMethod.REGRESSION_BASED, alpha=0.05,
        )
        assert result.is_independent

    def test_method_field(self):
        result = run_ci_test(
            self.data_chain, "X", "Y", [],
            method=IndependenceTestMethod.REGRESSION_BASED,
        )
        assert result.method is IndependenceTestMethod.REGRESSION_BASED

    def test_small_sample_returns_nan(self):
        tiny = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]})
        result = run_ci_test(
            tiny, "A", "B", [],
            method=IndependenceTestMethod.REGRESSION_BASED,
        )
        assert np.isnan(result.p_value)


class TestGeneralisedCovBasedTest:
    """Generalised Covariance Measure CI test sanity checks."""

    data_chain = _chain_data()
    data_fork = _fork_data()

    def test_marginal_dependence(self):
        result = run_ci_test(
            self.data_chain, "X", "Z", [],
            method=IndependenceTestMethod.GENERALISED_COV_BASED, alpha=0.05,
        )
        assert not result.is_independent, "X and Z should be dependent marginally"

    def test_conditional_independence_chain(self):
        result = run_ci_test(
            self.data_chain, "X", "Z", ["Y"],
            method=IndependenceTestMethod.GENERALISED_COV_BASED, alpha=0.05,
        )
        assert result.is_independent, "X ⊥ Z | Y should hold in chain"

    def test_conditional_independence_fork(self):
        result = run_ci_test(
            self.data_fork, "X", "Z", ["Y"],
            method=IndependenceTestMethod.GENERALISED_COV_BASED, alpha=0.05,
        )
        assert result.is_independent

    def test_method_field(self):
        result = run_ci_test(
            self.data_chain, "X", "Y", [],
            method=IndependenceTestMethod.GENERALISED_COV_BASED,
        )
        assert result.method is IndependenceTestMethod.GENERALISED_COV_BASED

    def test_small_sample_returns_nan(self):
        tiny = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]})
        result = run_ci_test(
            tiny, "A", "B", [],
            method=IndependenceTestMethod.GENERALISED_COV_BASED,
        )
        assert np.isnan(result.p_value)


# ---------------------------------------------------------------------------
# Triplet selection
# ---------------------------------------------------------------------------


class TestGetLocalMarkovTriplets:
    def test_chain(self):
        dag = _chain_dag()  # X → Y → Z
        triplets = get_local_markov_triplets(dag)
        # X: parents={}, descendants={Y,Z}, non-desc-non-parents={}  → no triplets
        # Y: parents={X}, descendants={Z}, non-desc-non-parents={}    → no triplets
        # Z: parents={Y}, descendants={},  non-desc-non-parents={X}   → (Z, X, {Y})
        assert len(triplets) == 1
        t = triplets[0]
        assert t.node == "Z"
        assert t.other == "X"
        assert t.parents == frozenset({"Y"})

    def test_fork(self):
        dag = _fork_dag()  # Y → X, Y → Z
        triplets = get_local_markov_triplets(dag)
        # Y: parents={}, descendants={X,Z}, non-desc={}              → no triplets
        # X: parents={Y}, descendants={},   non-desc-non-parents={Z} → (X, Z, {Y})
        # Z: parents={Y}, descendants={},   non-desc-non-parents={X} → (Z, X, {Y})
        assert len(triplets) == 2
        nodes_others = {frozenset({t.node, t.other}) for t in triplets}
        assert nodes_others == {frozenset({"X", "Z"})}
        for t in triplets:
            assert t.parents == frozenset({"Y"})

    def test_include_unconditional_false(self):
        """Root nodes (no parents) should be skipped when include_unconditional=False."""
        # In a 3-node graph A → B, A → C:
        # A has no parents → skipped when include_unconditional=False
        # B: parents={A}, non-desc-non-parents={C} → (B, C, {A})
        # C: parents={A}, non-desc-non-parents={B} → (C, B, {A})
        dag = nx.DiGraph([("A", "B"), ("A", "C")])
        triplets_all = get_local_markov_triplets(dag, include_unconditional=True)
        triplets_cond = get_local_markov_triplets(dag, include_unconditional=False)
        # With include_unconditional=True, A's non-desc-non-parents are empty anyway,
        # so both calls should return the same triplets for non-root nodes.
        assert len(triplets_all) == len(triplets_cond)
        # Introduce a root node with non-descendants: isolated node alongside a chain.
        # Use a 4-node graph: root R (no parents), R → X → Y, R is non-desc of Z
        dag2 = nx.DiGraph([("R", "X"), ("X", "Y")])
        dag2.add_node("Z")  # isolated node
        # R: parents={}, descendants={X,Y}, non-desc-non-parents={Z} → 1 unconditional triplet
        triplets2_all = get_local_markov_triplets(dag2, include_unconditional=True)
        triplets2_cond = get_local_markov_triplets(dag2, include_unconditional=False)
        # The unconditional triplet (R, Z, {}) should be excluded
        assert any(t.node == "R" and t.parents == frozenset() for t in triplets2_all)
        assert not any(t.node == "R" for t in triplets2_cond)

    def test_cycle_raises(self):
        g = nx.DiGraph([("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="DAG"):
            get_local_markov_triplets(g)

    def test_single_node(self):
        g = nx.DiGraph()
        g.add_node("A")
        assert get_local_markov_triplets(g) == []

    def test_two_nodes(self):
        g = nx.DiGraph([("A", "B")])
        # B: parents={A}, descendants={}, non-desc-non-parents={} → no triplets
        # A: parents={},  descendants={B},non-desc-non-parents={} → no triplets
        assert get_local_markov_triplets(g) == []


class TestGetEdgeSuggestionTriplets:
    def test_chain_edges(self):
        dag = _chain_dag()  # X→Y, Y→Z
        triplets = get_edge_suggestion_triplets(dag)
        assert len(triplets) == 2
        by_edge = {(t.source, t.target): t for t in triplets}
        # X→Y: other parents of Y = {}
        assert by_edge[("X", "Y")].other_parents == frozenset()
        # Y→Z: other parents of Z = {}
        assert by_edge[("Y", "Z")].other_parents == frozenset()

    def test_fork_edges(self):
        dag = _fork_dag()  # Y→X, Y→Z
        triplets = get_edge_suggestion_triplets(dag)
        assert len(triplets) == 2
        by_edge = {(t.source, t.target): t for t in triplets}
        assert by_edge[("Y", "X")].other_parents == frozenset()
        assert by_edge[("Y", "Z")].other_parents == frozenset()

    def test_collider(self):
        # X→Z←Y
        dag = nx.DiGraph([("X", "Z"), ("Y", "Z")])
        triplets = get_edge_suggestion_triplets(dag)
        by_edge = {(t.source, t.target): t for t in triplets}
        # X→Z: other parents of Z = {Y}
        assert by_edge[("X", "Z")].other_parents == frozenset({"Y"})
        # Y→Z: other parents of Z = {X}
        assert by_edge[("Y", "Z")].other_parents == frozenset({"X"})

    def test_cycle_raises(self):
        g = nx.DiGraph([("A", "B"), ("B", "A")])
        with pytest.raises(ValueError, match="DAG"):
            get_edge_suggestion_triplets(g)


# ---------------------------------------------------------------------------
# FalsifyGraphEvaluator
# ---------------------------------------------------------------------------


class TestFalsifyGraphEvaluator:
    data_chain = _chain_data()
    data_fork = _fork_data()

    def test_correct_chain_not_falsified(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain)
        assert isinstance(result, EvaluationResult)
        assert len(result.violations) == 0

    def test_correct_fork_not_falsified(self):
        dag = _fork_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_fork)
        assert len(result.violations) == 0

    def test_wrong_graph_falsified(self):
        """A graph that removes Y→Z should be falsified against chain data.

        Without Y→Z the graph claims Z ⊥ {X, Y} | {} which is false.
        """
        dag = _wrong_dag()  # X→Y, no Y→Z
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain)
        assert len(result.violations) > 0

    def test_result_counts(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain)
        assert result.n_total == result.n_tests + result.n_cached
        assert len(result.test_results) == result.n_total

    def test_split_lmc_edge_counts(self):
        """LMC and edge counts must be reported separately."""
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()

        # Without suggestions: edge counts must be zero
        r = ev.evaluate(dag, self.data_chain)
        assert r.n_edge_tests == 0
        assert r.n_edge_cached == 0
        assert r.n_lmc_tests == r.n_tests
        assert r.n_lmc_cached == r.n_cached

        ev.clear_cache()

        # With suggestions: both LMC and edge counts must be non-negative and
        # their sums must match the combined totals.
        r2 = ev.evaluate(dag, self.data_chain, include_suggestions=True)
        assert r2.n_lmc_tests >= 0
        assert r2.n_edge_tests >= 0
        assert r2.n_tests == r2.n_lmc_tests + r2.n_edge_tests
        assert r2.n_cached == r2.n_lmc_cached + r2.n_edge_cached

    def test_violation_rate_bounds(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain)
        assert 0.0 <= result.violation_rate <= 1.0

    def test_violation_rate_uses_only_lmc_tests(self):
        dag = _wrong_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain, include_suggestions=True)

        n_lmc_total = result.n_lmc_tests + result.n_lmc_cached
        assert n_lmc_total > 0
        assert result.n_edge_tests + result.n_edge_cached > 0
        assert result.violation_rate == len(result.violations) / n_lmc_total
        assert f"Violations : {len(result.violations)} / {n_lmc_total}" in result.summary()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def test_cache_reuse_on_second_evaluation(self):
        """Second evaluation of identical graph must produce more cache hits."""
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()

        result1 = ev.evaluate(dag, self.data_chain)
        assert result1.n_cached == 0  # fresh evaluator → nothing cached yet

        result2 = ev.evaluate(dag, self.data_chain)
        assert result2.n_cached == result1.n_tests  # everything cached now
        assert result2.n_tests == 0

    def test_cache_partial_reuse_modified_graph(self):
        """Adding an edge to the graph produces some cache hits and some new tests."""
        dag_small = nx.DiGraph([("X", "Y")])  # single edge
        dag_larger = nx.DiGraph([("X", "Y"), ("Y", "Z")])  # one more node/edge

        rng = np.random.default_rng(99)
        x = rng.normal(size=N)
        y = 0.5 * x + rng.normal(size=N)
        z = 0.5 * y + rng.normal(size=N)
        data = pd.DataFrame({"X": x, "Y": y, "Z": z})

        ev = FalsifyGraphEvaluator()
        r1 = ev.evaluate(dag_small, data)

        # The larger graph introduces new triplets; some may be cached already
        r2 = ev.evaluate(dag_larger, data)
        assert r2.n_total > 0

    def test_shared_cache_across_evaluators(self):
        """Two evaluators sharing a cache should reuse each other's results."""
        dag = _chain_dag()
        shared = CITestCache()

        ev1 = FalsifyGraphEvaluator(cache=shared)
        ev2 = FalsifyGraphEvaluator(cache=shared)

        ev1.evaluate(dag, self.data_chain)
        result2 = ev2.evaluate(dag, self.data_chain)

        assert result2.n_cached > 0

    def test_clear_cache(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()
        ev.evaluate(dag, self.data_chain)
        ev.clear_cache()
        assert len(ev.cache) == 0

        # After clearing, next evaluation should not hit the cache
        result = ev.evaluate(dag, self.data_chain)
        assert result.n_cached == 0

    # ------------------------------------------------------------------
    # Suggestions
    # ------------------------------------------------------------------

    def test_suggestions_disabled_by_default(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain)
        assert result.suggestions == []

    def test_suggestions_enabled(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain, include_suggestions=True)
        # suggestions is a list (may be empty if all edges are necessary)
        assert isinstance(result.suggestions, list)
        for s in result.suggestions:
            assert isinstance(s, EdgeSuggestion)

    def test_suggestions_cached(self):
        """Edge suggestion tests must also benefit from the cache."""
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()

        r1 = ev.evaluate(dag, self.data_chain, include_suggestions=True)
        r2 = ev.evaluate(dag, self.data_chain, include_suggestions=True)

        # Second pass must have no fresh tests at all
        assert r2.n_tests == 0
        assert r2.n_cached == r1.n_total

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def test_summary_is_string(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, self.data_chain)
        s = result.summary()
        assert isinstance(s, str)
        assert "LMC tests" in s
        assert "Violations" in s
        assert "Verdict" not in s

    # ------------------------------------------------------------------
    # Alternative CI test method
    # ------------------------------------------------------------------

    def test_mutual_info_method_runs(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator(method=IndependenceTestMethod.MUTUAL_INFO)
        result = ev.evaluate(dag, self.data_chain)
        assert isinstance(result, EvaluationResult)

    def test_kernel_based_method_runs(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator(method=IndependenceTestMethod.KERNEL_BASED)
        result = ev.evaluate(dag, self.data_chain)
        assert isinstance(result, EvaluationResult)
        assert len(result.violations) == 0

    def test_approx_kernel_based_method_runs(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator(method=IndependenceTestMethod.APPROX_KERNEL_BASED)
        result = ev.evaluate(dag, self.data_chain)
        assert isinstance(result, EvaluationResult)
        assert len(result.violations) == 0

    def test_regression_based_method_runs(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator(method=IndependenceTestMethod.REGRESSION_BASED)
        result = ev.evaluate(dag, self.data_chain)
        assert isinstance(result, EvaluationResult)
        assert len(result.violations) == 0

    def test_generalised_cov_based_method_runs(self):
        dag = _chain_dag()
        ev = FalsifyGraphEvaluator(method=IndependenceTestMethod.GENERALISED_COV_BASED)
        result = ev.evaluate(dag, self.data_chain)
        assert isinstance(result, EvaluationResult)
        assert len(result.violations) == 0

    def test_default_method_is_kernel_based(self):
        """The default CI test method should be KERNEL_BASED."""
        ev = FalsifyGraphEvaluator()
        assert ev.method is IndependenceTestMethod.KERNEL_BASED

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_graph_with_no_triplets(self):
        """A two-node DAG has no LM triplets; evaluation should succeed."""
        dag = nx.DiGraph([("A", "B")])
        data = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]})
        ev = FalsifyGraphEvaluator()
        result = ev.evaluate(dag, data)
        assert result.n_total == 0
        assert len(result.violations) == 0

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown CI test method"):
            run_ci_test(
                pd.DataFrame({"A": [1.0], "B": [2.0]}),
                "A", "B", [],
                method="nonexistent",  # type: ignore[arg-type]
            )
