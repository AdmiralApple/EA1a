"""
Hill climbing utilities"""

from __future__ import annotations
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from fitness import play_GPac
from histogram import RoundedFitnessHistogramMaker
from tree_genotype import ParseTree, TreeGenotype, TreeNode


@dataclass
class HillClimbResult:
    """
    Container for the results of a single hill-climb run"""

    best_solution: TreeGenotype
    best_score: float
    best_log: List[str]
    stairstep_data: List[float]
    histogram: RoundedFitnessHistogramMaker


def hill_climb_run(
    num_evaluations: int,
    config,
    *,
    rng: Optional[random.Random] = None,
) -> HillClimbResult:
    """
    Perform a hill-climb search over Pac-Man controllers"""

    if num_evaluations <= 0:
        raise ValueError("num_evaluations must be positive")

    hill_cfg = dict(config.get("hill_climb", {}))

    if rng is None:
        seed = hill_cfg.get("seed")
        rng = random.Random(seed) if seed is not None else random.Random()
    else:
        seed = hill_cfg.get("seed")
        if seed is not None:
            rng.seed(seed)

    if seed is not None:
        random.seed(seed)

    restart_interval = hill_cfg.get("restart_interval")
    constant_sigma = float(hill_cfg.get("constant_sigma", 0.5))

    pac_cfg = dict(config["pac_init"])
    depth_limit = int(pac_cfg["depth_limit"])
    terminals = tuple(pac_cfg["terminals"])
    nonterminals = tuple(pac_cfg["nonterminals"])
    max_mut_depth = int(hill_cfg.get("max_mutation_depth", depth_limit))
    max_mut_depth = max(0, min(max_mut_depth, depth_limit))

    histogram = RoundedFitnessHistogramMaker()
    stairstep_data: List[float] = []

    # Helper to evaluate a candidate controller and update global book keeping.
    def evaluate_candidate(individual: TreeGenotype) -> Tuple[float, List[str]]:
        score, log = play_GPac(individual, **config["game"])
        histogram.add(score)
        return score, log

    # Generate the initial solution and evaluate it immediately.
    current = TreeGenotype.initialization(1, **pac_cfg, rng=rng)[0]
    current_score, current_log = evaluate_candidate(current)

    best_solution = deepcopy(current)
    best_score = current_score
    best_log = list(current_log)
    stairstep_data.append(best_score)

    evaluations = 1
    since_restart = 1

    while evaluations < num_evaluations:
        if restart_interval and since_restart >= restart_interval:
            current = TreeGenotype.initialization(1, **pac_cfg, rng=rng)[0]
            current_score, current_log = evaluate_candidate(current)
            since_restart = 1
        else:
            candidate = deepcopy(current)
            mutate_tree_in_place(
                candidate.genes,
                depth_limit=depth_limit,
                max_mutation_depth=max_mut_depth,
                terminals=terminals,
                nonterminals=nonterminals,
                rng=rng,
                constant_sigma=constant_sigma,
            )
            candidate_score, candidate_log = evaluate_candidate(candidate)
            if candidate_score > current_score:
                current = candidate
                current_score = candidate_score
                current_log = candidate_log
            since_restart += 1

        evaluations += 1

        if current_score > best_score:
            best_solution = deepcopy(current)
            best_score = current_score
            best_log = list(current_log)

        stairstep_data.append(best_score)

    return HillClimbResult(
        best_solution=best_solution,
        best_score=best_score,
        best_log=best_log,
        stairstep_data=stairstep_data,
        histogram=histogram,
    )


def hill_climb_experiment(
    num_runs: int,
    num_evaluations: int,
    config,
    *,
    base_seed: Optional[int] = None,
) -> Tuple[List[float], TreeGenotype, List[str], List[float], RoundedFitnessHistogramMaker]:
    """Execute multiple hill-climb runs and collate their outputs.

    The return signature mirrors ``random_search_experiment`` from the
    notebook so that the same downstream utilities can be reused when
    generating plots or reports.
    """

    if num_runs <= 0:
        raise ValueError("num_runs must be positive")

    best_per_run: List[float] = []
    best_solution: Optional[TreeGenotype] = None
    best_log: Optional[List[str]] = None
    best_score = -math.inf
    stairstep_data: Optional[List[float]] = None
    histograms: List[RoundedFitnessHistogramMaker] = []

    for run_index in range(num_runs):
        run_config = {
            key: (value.copy() if hasattr(value, "copy") else value)
            for key, value in config.items()
        }
        hill_cfg = dict(run_config.get("hill_climb", {}))
        if base_seed is not None:
            hill_cfg["seed"] = base_seed + run_index
        run_config["hill_climb"] = hill_cfg

        result = hill_climb_run(num_evaluations, run_config)
        histograms.append(result.histogram)
        best_per_run.append(result.best_score)

        if result.best_score > best_score:
            best_score = result.best_score
            best_solution = result.best_solution
            best_log = result.best_log
            stairstep_data = result.stairstep_data

    merged_hist = RoundedFitnessHistogramMaker.merge(histograms)

    assert best_solution is not None and best_log is not None and stairstep_data is not None

    return best_per_run, best_solution, best_log, stairstep_data, merged_hist


def mutate_tree_in_place(
    tree: ParseTree,
    *,
    depth_limit: int,
    max_mutation_depth: int,
    terminals: Iterable[str],
    nonterminals: Iterable[str],
    rng: random.Random,
    constant_sigma: float,
) -> None:
    """Apply a small random modification to ``tree``.

    The mutation keeps the tree depth within ``depth_limit`` by replacing
    the selected subtree with a freshly generated structure when needed.
    Additional neighbourhood operators perturb numeric terminals,
    permute children, or change nonterminal primitives to encourage a
    diverse local search.
    """

    if tree.root is None:
        raise ValueError("tree must define a root before mutation")

    terminals = tuple(terminals)
    nonterminals = tuple(nonterminals)

    target, parent, child_index, depth = _sample_node(tree.root, rng)

    if target is None:
        return

    operations = ["replace"]

    if target.primitive == "C":
        operations.append("perturb_constant")
    if target.primitive in nonterminals and len(target.children) == 2:
        operations.extend(["swap_children", "mutate_operator"])

    operation = rng.choice(operations)

    if operation == "replace":
        max_depth = min(depth_limit - depth, max_mutation_depth)
        new_subtree = _build_random_subtree(
            max_depth,
            terminals=terminals,
            nonterminals=nonterminals,
            rng=rng,
        )
        if parent is None:
            tree.root = new_subtree
        else:
            parent.children[child_index] = new_subtree
        return

    if operation == "perturb_constant":
        low, high = TreeNode.CONSTANT_RANGE
        offset = rng.normalvariate(0.0, constant_sigma)
        target.value = max(low, min(high, float(target.value) + offset))
        return

    if operation == "swap_children":
        target.children[0], target.children[1] = target.children[1], target.children[0]
        return

    if operation == "mutate_operator":
        alternatives = [op for op in nonterminals if op != target.primitive]
        if not alternatives:
            return
        target.primitive = rng.choice(alternatives)
        return


def _sample_node(
    root: TreeNode,
    rng: random.Random,
) -> Tuple[Optional[TreeNode], Optional[TreeNode], int, int]:
    """Return a uniformly sampled node with its parent metadata."""

    stack: List[Tuple[TreeNode, Optional[TreeNode], int, int]] = [(root, None, 0, 0)]
    nodes: List[Tuple[TreeNode, Optional[TreeNode], int, int]] = []

    while stack:
        node, parent, index, depth = stack.pop()
        nodes.append((node, parent, index, depth))
        for child_index, child in enumerate(node.children):
            stack.append((child, node, child_index, depth + 1))

    if not nodes:
        return None, None, 0, 0

    return rng.choice(nodes)


def _build_random_subtree(
    max_depth: int,
    *,
    terminals: Tuple[str, ...],
    nonterminals: Tuple[str, ...],
    rng: random.Random,
) -> TreeNode:
    """Generate a subtree whose depth does not exceed ``max_depth``."""

    max_depth = max(0, max_depth)

    if max_depth == 0 or not nonterminals:
        primitive = rng.choice(terminals)
        return TreeNode.make_terminal(primitive, rng=rng)

    if rng.random() < 0.5:
        subtree = TreeGenotype.generate_full_tree(
            max_depth,
            terminals=terminals,
            nonterminals=nonterminals,
            rng=rng,
        )
    else:
        subtree = TreeGenotype.generate_grow_tree(
            max_depth,
            terminals=terminals,
            nonterminals=nonterminals,
            rng=rng,
        )

    return deepcopy(subtree.root)