# Population Initialization with Ramped Half-And-Half

This guide explains how to execute and interpret the "Population Initialization with Ramped Half-And-Half" section of `2a_notebook.ipynb` after implementing `TreeGenotype.initialization`.

## Running the Notebook Section
1. Open `2a_notebook.ipynb` in Jupyter Lab or Jupyter Notebook.
2. Run the earlier sections that set up configuration data and helper functions. These cells populate `config['pac_init']` with the `depth_limit`, `terminals`, and `nonterminals` parameters used by the initialization routine.
3. Execute the cell immediately under the "Population Initialization with Ramped Half-And-Half" heading. It contains the following code:
   ```python
   mu = 10
   example_population = TreeGenotype.initialization(mu, **config['pac_init'])
   display_population(example_population)
   ```
4. The helper `display_population` (defined in a prior cell) will print each individual's tree depth and structure. Verify that roughly half the individuals show strictly full trees while the others have varied shapes created by the grow method.

## What to Expect
- Tree depths will span the entire `[1, depth_limit]` range because the initialization call samples depths uniformly at random for every individual.
- The displayed tree listings should alternate between balanced (full) trees and irregular (grow) trees. When the population size is odd, the extra individual will be produced by the grow method, matching the expected half-and-half split.
- Rerunning the cell should reshuffle the depth distribution and the constant terminals because the procedure uses a random number generator.

## Troubleshooting Tips
- If you encounter `ValueError` messages about missing primitives or invalid depth limits, make sure the configuration cell defining `terminals`, `nonterminals`, and `depth_limit` was executed before calling `TreeGenotype.initialization`.
- To reproduce identical populations for debugging, pass a deterministic random-number generator in the notebook before the call:
  ```python
  rng = random.Random(0)
  example_population = TreeGenotype.initialization(mu, rng=rng, **config['pac_init'])
  ```
- Confirm that both `TreeGenotype.generate_full_tree` and `TreeGenotype.generate_grow_tree` from earlier notebook exercises execute without errors, since the ramped routine delegates to these helpers.

With these steps, the notebook cell will showcase a correctly ramped half-and-half population that is ready for subsequent evaluation and evolution experiments.
