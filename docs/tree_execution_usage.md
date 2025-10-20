# Executing GPac Parse Trees in 2a Notebook

## Overview
The parse tree genotype now includes an `evaluate` method that accepts the
observation dictionaries produced by `GPacGame.get_observations`. Terminal
primitive values are cached per state and the nonterminal operators support the
full primitive set from the notebook, including the safeguarded division and
`RAND` operator.

## Using Tree Execution in the Notebook
1. **Instantiate a controller** – Build or load a tree using any of the
   initialization helpers and assign it to a `TreeGenotype` instance, for
   example:
   ```python
   pac_genotype = TreeGenotype()
   pac_genotype.genes = TreeGenotype.generate_full_tree(
       depth_limit=2,
       terminals=config['pac_init']['terminals'],
       nonterminals=config['pac_init']['nonterminals']
   )
   ```
2. **Run the Tree Execution cell** – In the "Tree Execution" section, provide
   the genotype (or its `genes` object) as `pac_controller` before calling
   `play_GPac`:
   ```python
   pac_controller = pac_genotype
   score, log = play_GPac(pac_controller, **config['game'])
   ```
   The helper automatically evaluates each candidate state, selects the action
   with the highest score, and plays a full game.
3. **Inspect the outcome** – Use the returned `score` and `log` (or the
   existing visualization utilities in the notebook) to analyse the gameplay
   and iterate on your primitive definitions.

## Tips
- The controller can also be set to `pac_genotype.genes` directly if you only
  want to expose the `ParseTree` object; both provide the required `evaluate`
  method.
- Supply a seeded random number generator via `config['game']['rng']` when you
  need deterministic behaviour from the `RAND` primitive during debugging.
- The terminal cache is reset for each evaluated state, so feel free to reuse
  the same genotype across multiple games without manual cleanup.
