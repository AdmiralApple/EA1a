# Hill Climber Experiment Guide

This guide explains how to run the hill-climbing experiment that satisfies the RED 2 deliverable.

## 1. Load the configuration
Use the helper already provided in the notebook to read the new configuration file:

```python
config = read_config('configs/2a/hill_climber_config.txt', globals(), locals())
```

The configuration mirrors the structure of the green random-search configuration, but adds a `[hill_climb]` section. You can tweak the restart interval, constant perturbation variance, or subtree depth budget here before rerunning experiments.

## 2. Run a single hill-climb search
Import the helper and execute a single run to verify everything works:

```python
from hill_climber import hill_climb_run

result = hill_climb_run(10_000, config)
print('Best score:', result.best_score)
print('Serialized controller:')
print(result.best_solution.serialize())
```

The returned `HillClimbResult` exposes the best controller, the corresponding game log, the stairstep data, and a histogram you can reuse for plotting.

## 3. Launch the full experiment
For the full bonus experiment—matching what the notebook does for random search—call `hill_climb_experiment` and reuse the existing saving utilities:

```python
from hill_climber import hill_climb_experiment

(
    best_per_run,
    best_solution,
    best_log,
    stairstep_data,
    hist,
) = hill_climb_experiment(10, 10_000, config)

save_data(
    best_per_run,
    best_solution,
    best_log,
    stairstep_data,
    hist,
    Path('./data/2a/hill_climber/'),
    config,
)
```

The returned values match the interface of `random_search_experiment`, so you can reuse the same plotting code to generate the histogram and stairstep chart for your report.

## 4. Statistical comparison
Store the `best_per_run` scores in `data/2a/hill_climber/best_per_run.txt`, then compare them to the green experiment with `stats.py`:

```bash
python stats.py data/2a/green/best_per_run.txt data/2a/hill_climber/best_per_run.txt
```

This prints the descriptive statistics and Welch's t-test p-value required for the analysis section of the report.

## 5. Optional reproducibility controls
If you need deterministic runs, set `seed = <integer>` under `[hill_climb]` in the configuration file. The helper will use that seed for both tree mutations and the underlying GPac simulator.

These steps supply every artefact the RED 2 deliverable requests without modifying the notebook itself.
