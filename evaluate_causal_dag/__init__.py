"""
evaluate_causal_dag — multi-dimensional causal DAG accuracy evaluation.

This package provides a growing collection of methods for evaluating whether a
proposed causal directed acyclic graph (DAG) is consistent with observed data.

Quick start
-----------
>>> import networkx as nx, pandas as pd, numpy as np
>>> from evaluate_causal_dag.methods.falsify import FalsifyGraphEvaluator
>>>
>>> rng = np.random.default_rng(42)
>>> n = 500
>>> x = rng.normal(size=n)
>>> y = 0.8 * x + rng.normal(size=n)
>>> z = 0.6 * y + rng.normal(size=n)
>>> data = pd.DataFrame({"X": x, "Y": y, "Z": z})
>>>
>>> dag = nx.DiGraph([("X", "Y"), ("Y", "Z")])
>>> evaluator = FalsifyGraphEvaluator()
>>> result = evaluator.evaluate(dag, data)
>>> print(result.summary())

Available modules
-----------------
evaluate_causal_dag.methods.falsify
    Local-Markov-Condition based evaluation with result caching.
"""

from evaluate_causal_dag import methods

__all__ = ["methods"]
