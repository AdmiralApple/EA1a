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

    if "depth_limit" not in ghost_kwargs:
        raise KeyError("ghost_init.depth_limit must be defined")

    histogram = RoundedFitnessHistogramMaker()
    best_solution: TreeGenotype | None = None
    best_log: List[str] | None = None
    stairstep_data: List[float] = []
    best_score = math.inf

    for _ in range(num_evaluations):
        # Ramped half-and-half with mu=1 gives us either a grow or full tree.
        candidate = TreeGenotype.initialization(1, **ghost_kwargs)[0]

        score, log = play_GPac(
            pac_controller=None,
            ghost_controller=candidate,
            ghost_policy="min",
            **game_kwargs,
        )

        histogram.add(score)
        candidate.fitness = score

        if score < best_score:
            best_score = score
            best_solution = candidate
            best_log = list(log)

        stairstep_data.append(best_score)

    if best_solution is None or best_log is None:
        raise RuntimeError("Random search did not produce any candidates")

    return GhostRandomSearchResult(best_solution, best_score, best_log, stairstep_data, histogram)


def _ghost_multiprocess_helper(num_evaluations: int, config: dict) -> Tuple[str, float, List[str], List[float], RoundedFitnessHistogramMaker]:
    """Worker helper that serialises the best solution for safe cross-process transfer."""

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
    """Execute many ghost random-search runs and keep the strongest ghost controller."""

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

    with multiprocessing.Pool(processes=processes) as pool:
        args = [(num_evaluations, config)] * num_runs
        run_results = list(pool.starmap(_ghost_multiprocess_helper, args))

    for serialised, run_score, run_log, run_stairstep, histogram in run_results:
        controller = TreeGenotype()
        controller.deserialize(serialised)

        best_per_run.append(run_score)
        histograms.append(histogram)

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
    """Persist the collected experiment artifacts to disk for later analysis."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "best_per_run.txt").open("w", encoding="utf-8") as handle:
        for score in best_per_run:
            handle.write(f"{score}\n")

    with (output_dir / "best_solution.txt").open("w", encoding="utf-8") as handle:
        handle.write(best_solution.serialize())

    with (output_dir / "best_log.txt").open("w", encoding="utf-8") as handle:
        for line in best_log:
            handle.write(f"{line}\n")

    with (output_dir / "stairstep.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(list(stairstep_data)))

    histogram.save_to_file(output_dir / "histogram.txt")


def load_red1_config(path: str = "configs/2a/red1_config.txt") -> dict:
    """Load the Red 1 configuration using the same helper as the notebook."""

    return read_config(path, globals(), locals())


def _build_cli_parser() -> argparse.ArgumentParser:
    """Create an argument parser for convenient command-line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/2a/red1_config.txt", help="Config file path")
    parser.add_argument("--runs", type=int, default=10, help="Number of random-search runs to execute")
    parser.add_argument(
        "--evaluations", type=int, default=10_000, help="Evaluations per random-search run"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Optional override for multiprocessing worker count",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/2a/red1"),
        help="Directory for the saved experiment artifacts",
    )
    return parser


def main() -> None:
    """Entry-point that mirrors the final notebook cell for the Red 1 bonus experiment."""

    parser = _build_cli_parser()
    args = parser.parse_args()

    config = load_red1_config(args.config)

    (
        best_per_run,
        best_solution,
        best_log,
        stairstep_data,
        histogram,
    ) = ghost_random_search_experiment(
        args.runs,
        args.evaluations,
        config,
        processes=args.processes,
    )

    save_ghost_data(best_per_run, best_solution, best_log, stairstep_data, histogram, args.output_dir)

    figure = histogram.get_plot(
        f"Ghost Random Search over {args.runs} runs * {args.evaluations} evaluations"
    )
    figure.savefig(args.output_dir / "histogram.png")


if __name__ == "__main__":
    main()
