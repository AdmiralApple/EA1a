# Using Ramped Half-and-Half Initialization in 2a Notebook

## Overview
The `TreeGenotype.initialization` class method now creates populations with the ramped half-and-half strategy. Every call builds half of the individuals with the full method and the other half with the grow method while sampling maximum depths uniformly from the configuration range.

## How to Use in the Notebook
1. **Load the configuration** – Execute the notebook cell that loads `config['pac_init']`. This cell provides the `depth_limit`, `terminals`, and `nonterminals` entries required by the initialization method.
2. **Call the initializer** – In the "Population Initialization with Ramped Half-And-Half" section, run the provided cell (or create a new one) that calls:
   ```python
   example_population = TreeGenotype.initialization(mu, **config['pac_init'])
   ```
   Replace `mu` with the desired population size. The notebook cells later in the section already demonstrate values such as 10, 100, or 5; feel free to experiment with any positive integer.
3. **Inspect the results** – Iterate over `example_population` and print or visualize each individual's `genes` to confirm the mixture of full and grow trees. The existing helper functions in the notebook can assist with tree visualization.
4. **Adjust parameters if needed** – To explore deeper or shallower trees, edit `depth_limit` inside `configs/2a/green_config.txt` before re-running the relevant notebook cells. Increasing the limit allows more complex structures, whereas decreasing it constrains depth (keep it at least the default value as suggested in the notebook).

## Tips
- The initializer accepts an optional random number generator via the `rng` key in `config['pac_init']` if you wish to use a seeded generator for reproducibility.
- When testing multiple population sizes, rerun the initialization cell each time so the new parameters take effect immediately.
