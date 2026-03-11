"""
falsify — causal graph evaluation via Local Markov Condition testing.

Public API
----------
FalsifyGraphEvaluator
    Main evaluator class.
EvaluationResult
    Result object returned by :meth:`FalsifyGraphEvaluator.evaluate`.
EdgeSuggestion
    Suggestion to review a specific edge.
CITestResult
    Result of a single conditional independence test.
CITestCache
    Cache for CI-test results (shared across evaluations).
IndependenceTestMethod
    Enum of available CI test methods.
run_ci_test
    Low-level dispatcher for a single CI test.
get_local_markov_triplets
    Extract Local-Markov triplets from a DAG.
get_edge_suggestion_triplets
    Extract edge-suggestion triplets from a DAG.
"""

from evaluate_causal_dag.methods.falsify.cache import CITestCache
from evaluate_causal_dag.methods.falsify.evaluator import (
    EdgeSuggestion,
    EvaluationResult,
    FalsifyGraphEvaluator,
)
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

__all__ = [
    "FalsifyGraphEvaluator",
    "EvaluationResult",
    "EdgeSuggestion",
    "CITestResult",
    "CITestCache",
    "IndependenceTestMethod",
    "run_ci_test",
    "LocalMarkovTriplet",
    "EdgeSuggestionTriplet",
    "get_local_markov_triplets",
    "get_edge_suggestion_triplets",
]
