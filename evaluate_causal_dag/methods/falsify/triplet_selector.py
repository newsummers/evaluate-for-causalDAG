"""
Triplet selection from a directed acyclic graph (DAG).

Two families of triplets are produced:

Local Markov triplets
    For every node *X* the **Local Markov Condition** (LMC) states that *X* is
    conditionally independent of all its non-descendants given its parents:

        X ⊥ NonDesc(X) \\ Pa(X)  |  Pa(X)

    A violation of the LMC for any node suggests that the DAG is mis-specified.

Edge-suggestion triplets
    For every directed edge *X → Y* the edge is necessary only if *X* is
    **not** independent of *Y* given the other parents of *Y*:

        X ⊬ Y  |  Pa(Y) \\ {X}

    If the independence test says *X ⊥ Y | Pa(Y) \\ {X}* the edge *X → Y*
    may be superfluous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocalMarkovTriplet:
    """A single Local-Markov independence statement implied by a DAG.

    Attributes
    ----------
    node:
        The node whose Markov condition we are testing.
    other:
        One non-descendant, non-parent node that should be independent of
        *node* given *parents*.
    parents:
        Frozenset of parents of *node* (the conditioning set).
    """

    node: str
    other: str
    parents: FrozenSet[str]


@dataclass(frozen=True)
class EdgeSuggestionTriplet:
    """A triplet derived from a single edge *source → target*.

    The triplet encodes the test:  source ⊥ target | (Pa(target) \\ {source})

    If the test returns *independent* the edge may be superfluous.
    """

    source: str
    target: str
    other_parents: FrozenSet[str]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_local_markov_triplets(dag: nx.DiGraph) -> List[LocalMarkovTriplet]:
    """Return all Local-Markov independence statements implied by *dag*.

    For every node *X* and every non-descendant, non-parent node *Z* the
    function produces a :class:`LocalMarkovTriplet` (X, Z, Pa(X)).

    Parameters
    ----------
    dag:
        A directed acyclic graph represented as :class:`networkx.DiGraph`.

    Returns
    -------
    list of LocalMarkovTriplet
        One entry per (node, non-desc-non-parent) pair.  The list is empty
        for graphs with fewer than two nodes.

    Raises
    ------
    ValueError
        If *dag* contains a cycle.
    """
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The input graph must be a DAG (no cycles).")

    triplets: List[LocalMarkovTriplet] = []
    nodes = list(dag.nodes())

    for node in nodes:
        parents: FrozenSet[str] = frozenset(dag.predecessors(node))
        descendants: set = nx.descendants(dag, node)

        for other in nodes:
            if other == node:
                continue
            if other in descendants:
                continue
            if other in parents:
                continue
            triplets.append(LocalMarkovTriplet(node=node, other=other, parents=parents))

    return triplets


def get_edge_suggestion_triplets(dag: nx.DiGraph) -> List[EdgeSuggestionTriplet]:
    """Return one triplet per directed edge in *dag*.

    For the edge *X → Y* the triplet encodes whether *X* is still needed as a
    parent of *Y* given all other parents of *Y*.

    Parameters
    ----------
    dag:
        A directed acyclic graph.

    Returns
    -------
    list of EdgeSuggestionTriplet
        One entry per directed edge.
    """
    if not nx.is_directed_acyclic_graph(dag):
        raise ValueError("The input graph must be a DAG (no cycles).")

    triplets: List[EdgeSuggestionTriplet] = []
    for source, target in dag.edges():
        other_parents: FrozenSet[str] = frozenset(
            p for p in dag.predecessors(target) if p != source
        )
        triplets.append(
            EdgeSuggestionTriplet(
                source=source,
                target=target,
                other_parents=other_parents,
            )
        )
    return triplets
