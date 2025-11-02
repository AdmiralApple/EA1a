"""
Ghost random search experiment helpers for 2a Red 1"""

from __future__ import annotations
import argparse
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
from histogram import RoundedFitnessHistogramMaker
from snake_eyes import read_config
from fitness import play_GPac
from tree_genotype import TreeGenotype


@dataclass
class GhostRandomSearchResult:
    """
    Container capturing the artifacts from a single run."""

    best_solution: TreeGenotype
    best_score: float
    best_log: List[str]
    stairstep_data: List[float]
    histogram: RoundedFitnessHistogramMaker


def _validate_config_section(config: dict, section: str) -> dict:
    """
    error validation."""

    if section not in config:
        raise KeyError(
            f"Configuration is missing the required '[{section}]' section for Red 1."
        )
    return dict(config[section])


def ghost_random_search_run(num_evaluations: int, config: dict) -> GhostRandomSearchResult:
    """
    Execute a single random sweep that minimises Pac-Man's score."""

    ghost_kwargs = _validate_config_section(config, "ghost_init")
    game_kwargs = dict(config.get("game", {}))

    # Require a depth limit to keep randomly generated trees bounded.
    if "depth_limit" not in ghost_kwargs:
        raise KeyError("ghost_init.depth_limit must be defined")

    # Track aggregate stats and the best-so-far controller.
    histogram = RoundedFitnessHistogramMaker()
    best_solution: TreeGenotype | None = None
    best_log: List[str] | None = None
    stairstep_data: List[float] = []
    best_score = math.inf

    for _ in range(num_evaluations):
        # Ramped half-and-half with mu=1 gives us either a grow or full tree.
        candidate = TreeGenotype.initialization(1, **ghost_kwargs)[0]

        # Play a single game with this ghost controller; use "min" policy so
        # ghosts attempt to reduce Pac-Man's score.
        score, log = play_GPac(
            pac_controller=None,
            ghost_controller=candidate,
            ghost_policy="min",
            **game_kwargs,
        )

        histogram.add(score)
        candidate.fitness = score

        # Update best-so-far if this candidate improves the minimum score.
        if score < best_score:
            best_score = score
            best_solution = candidate
            best_log = list(log)

        stairstep_data.append(best_score)

    # Sanity check: ensure at least one candidate was produced and tracked.
    if best_solution is None or best_log is None:
        raise RuntimeError("Random search did not produce any candidates")

    # Package the run artifacts into a structured result container
    return GhostRandomSearchResult(best_solution, best_score, best_log, stairstep_data, histogram)


def _ghost_multiprocess_helper(num_evaluations: int, config: dict) -> Tuple[str, float, List[str], List[float], RoundedFitnessHistogramMaker]:
    """
    Worker helper that serialises the best solution for safe cross-process use."""

    # Run a single search in this worker and serialise the best controller.
    result = ghost_random_search_run(num_evaluations, config)
    return (
        result.best_solution.serialize(),
        result.best_score,
        result.best_log,
        result.stairstep_data,
        result.histogram,
    )


def ghost_random_search_experiment(
    num_runs: int,
    num_evaluations: int,
    config: dict,
    *,
    processes: int | None = None,
) -> Tuple[List[float], TreeGenotype, List[str], List[float], RoundedFitnessHistogramMaker]:
    """
    Execute many ghost random-search runs and keep the strongest ghost controller."""

    if num_runs < 1:
        raise ValueError("num_runs must be at least 1")
    if num_evaluations < 1:
        raise ValueError("num_evaluations must be at least 1")

    best_overall_score = math.inf
    best_solution: TreeGenotype | None = None
    best_log: List[str] | None = None
    stairstep_data: List[float] | None = None

    best_per_run: List[float] = []
    histograms: List[RoundedFitnessHistogramMaker] = []

    # Use a process pool to parallelise independent runs for speed.
    with multiprocessing.Pool(processes=processes) as pool:
        args = [(num_evaluations, config)] * num_runs
        run_results = list(pool.starmap(_ghost_multiprocess_helper, args))

    for serialised, run_score, run_log, run_stairstep, histogram in run_results:
        controller = TreeGenotype()
        controller.deserialize(serialised)

        best_per_run.append(run_score)
        histograms.append(histogram)

        # Update global best if this run produced a stronger controller.
        if run_score < best_overall_score:
            best_overall_score = run_score
            best_solution = controller
            best_log = run_log
            stairstep_data = run_stairstep

    if best_solution is None or best_log is None or stairstep_data is None:
        raise RuntimeError("Experiment failed to capture a best controller")

    merged_hist = RoundedFitnessHistogramMaker.merge(histograms)
    return best_per_run, best_solution, best_log, stairstep_data, merged_hist


def save_ghost_data(
    best_per_run: Sequence[float],
    best_solution: TreeGenotype,
    best_log: Sequence[str],
    stairstep_data: Sequence[float],
    histogram: RoundedFitnessHistogramMaker,
    output_dir: Path,
) -> None:
    """
    Persist experiment artifacts to disk for later analysis"""

    # Ensure the target directory exists so subsequent writes succeed.
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "best_per_run.txt").open("w", encoding="utf-8") as handle:
        for score in best_per_run:
            handle.write(f"{score}\n")

    # Serialise the tree genotype
    with (output_dir / "best_solution.txt").open("w", encoding="utf-8") as handle:
        handle.write(best_solution.serialize())

    # Persist the game log for the best run
    with (output_dir / "best_log.txt").open("w", encoding="utf-8") as handle:
        for line in best_log:
            handle.write(f"{line}\n")

    # Save the running best scores
    with (output_dir / "stairstep.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(list(stairstep_data)))

    histogram.save_to_file(output_dir / "histogram.txt")

