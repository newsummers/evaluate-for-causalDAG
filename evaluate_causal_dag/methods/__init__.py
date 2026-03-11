"""
evaluate_causal_dag.methods — collection of causal-graph evaluation methods.

Available methods
-----------------
falsify
    Evaluate a DAG by testing the Local Markov Condition against observed data.
    Inspired by the ``falsify_graph`` approach; features result caching for
    efficient re-evaluation of similar graphs.
"""

from evaluate_causal_dag.methods import falsify

__all__ = ["falsify"]
