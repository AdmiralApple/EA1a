# Running Random Search in the 2a Notebook

## Overview
The `random_search_run` helper evaluates randomly generated `TreeGenotype`
controllers and keeps the best-performing candidate along with supporting
statistics. Each evaluation samples a depth between one and the configured
maximum, flips a fair coin between the full and grow construction methods, and
plays a complete GPac game using the notebook's game configuration. The
function also records histogram counts and a stairstep trace so you can build
the plots requested later in the notebook.

## How to Use It in the Notebook
1. **Import the helper** – After executing the cell that installs dependencies
   and loads `read_config`, run a new code cell containing:
   ```python
   from random_search import random_search_run
   ```
   This overrides the placeholder definition provided in the notebook without
   modifying the notebook itself.
2. **Load the configuration** – Execute the existing cell
   ```python
   config = read_config('configs/2a/green_config.txt', globals(), locals())
   ```
   so that both the `pac_init` and `game` sections are available.
3. **Run a single search** – In the "Random Search Algorithm" section, replace
   the call to the stub with:
   ```python
   results = random_search_run(num_evaluations, config)
   (
       best_solution,
       best_score,
       best_log,
       stairstep_data,
       hist,
   ) = results
   ```
   The function returns the best individual, its score, the corresponding game
   log, the running-best stairstep data, and a populated histogram maker.
4. **Proceed with the experiment cells** – Downstream cells (e.g.,
   `random_search_experiment`, histogram plotting, and stairstep generation)
   work unchanged because they already expect the structure returned by the
   helper.

## Data Collected per Run
- `best_solution`: The `TreeGenotype` instance that achieved the highest score.
- `best_score`: The numeric GPac score produced by `best_solution`.
- `best_log`: The gameplay log from the best-scoring evaluation (ready for
  `render_game`).
- `stairstep_data`: A list whose `i`‑th entry equals the best score seen after
  evaluation `i+1`, enabling stairstep visualisations.
- `hist`: The `RoundedFitnessHistogramMaker` containing all sampled scores for
  quick plotting and merging across multiple runs.
