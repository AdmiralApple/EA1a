"""Utilities for performing random search runs in the GPac assignment."""
from __future__ import annotations

import random
from math import inf
from typing import Any, Dict, List, Tuple

from fitness import play_GPac
from histogram import RoundedFitnessHistogramMaker
from tree_genotype import TreeGenotype


def _get_rng(pac_config: Dict[str, Any]) -> random.Random:
    """Return the RNG instance to use for sampling trees."""
    rng = pac_config.get("rng", random)
    return rng


def _build_random_individual(pac_config: Dict[str, Any], rng: random.Random) -> TreeGenotype:
    """Create a single individual by sampling depth and construction strategy."""
    depth_limit = pac_config.get("depth_limit")
    terminals: Tuple[str, ...] = tuple(pac_config.get("terminals", ()))
    nonterminals: Tuple[str, ...] = tuple(pac_config.get("nonterminals", ()))

    if depth_limit is None:
        raise KeyError("pac_config must provide a 'depth_limit' entry")
    if not terminals:
        raise ValueError("pac_config must provide at least one terminal primitive")
    if not nonterminals:
        raise ValueError("pac_config must provide at least one nonterminal primitive")

    # Sample a maximum depth uniformly so we mimic ramped half-and-half behaviour.
    sampled_depth = rng.randint(1, int(depth_limit))
    use_full_method = rng.random() < 0.5

    if use_full_method:
        tree = TreeGenotype.generate_full_tree(
            sampled_depth,
            terminals=terminals,
            nonterminals=nonterminals,
            rng=rng,
        )
    else:
        tree = TreeGenotype.generate_grow_tree(
            sampled_depth,
            terminals=terminals,
            nonterminals=nonterminals,
            rng=rng,
        )

    individual = TreeGenotype()
    individual.genes = tree
    return individual


def random_search_run(num_evaluations: int, config: Dict[str, Any]):
    """Evaluate ``num_evaluations`` random controllers and track their scores.

    Parameters
    ----------
    num_evaluations:
        Number of candidate controllers to sample and evaluate.
    config:
        Assignment configuration dictionary produced by ``read_config``.  The
        ``pac_init`` section supplies tree-construction parameters, while the
        ``game`` section is forwarded to ``play_GPac``.

    Returns
    -------
    Tuple containing the best solution, its score, the associated game log,
    stairstep progression data, and the populated histogram.
    """

    if num_evaluations <= 0:
        raise ValueError("num_evaluations must be positive")

    pac_config = config.get("pac_init")
    if pac_config is None:
        raise KeyError("config must define a 'pac_init' section")

    game_config = config.get("game")
    if game_config is None:
        raise KeyError("config must define a 'game' section")

    hist = RoundedFitnessHistogramMaker()
    best_score = -inf
    best_solution: TreeGenotype | None = None
    best_log: List[str] | None = None
    stairstep_data: List[float] = []

    rng = _get_rng(pac_config)

    for _ in range(num_evaluations):
        # Sample a brand-new controller for every evaluation.
        candidate = _build_random_individual(pac_config, rng)
        score, log = play_GPac(candidate, **game_config)

        hist.add(score)

        if score > best_score:
            best_score = score
            best_solution = candidate
            # Store the log immediately to avoid retaining logs for inferior runs.
            best_log = list(log)

        # Record the running best to support stairstep visualisations.
        stairstep_data.append(best_score)

    if best_solution is None or best_log is None:
        # This should only happen if num_evaluations is zero, which is already guarded above.
        raise RuntimeError("random search failed to evaluate any solutions")

    return best_solution, best_score, best_log, stairstep_data, hist
