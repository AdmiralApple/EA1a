# Tree Execution in Assignment 2a

## Overview
The `ParseTree.evaluate` helper executes GPac trees by recursively visiting each node and
combining child values with the configured primitive. Terminal primitives cache their values
during an evaluation, so repeated sensors such as `P` or `G` are only computed once per
state. The public `TreeGenotype.evaluate` method simply forwards to the stored parse tree,
providing a consistent interface whether the notebook stores an entire genotype or the tree
itself.

## Using the Implementation in the Notebook
1. **Load the configuration** – Run the notebook cell that reads `configs/2a/green_config.txt`
   so you have access to the `terminals`, `nonterminals`, and depth limit parameters.
2. **Create or load a tree** – Build a controller using your preferred initialization strategy.
   For example, `TreeGenotype.generate_grow_tree` returns a `ParseTree` ready for execution.
   Assign the tree to a genotype with `individual.genes = tree` if you plan to reuse the
   provided `TreeGenotype` wrapper.
3. **Assign the controller** – In the "Tree Execution" section, set `pac_controller` to either
   the `TreeGenotype` instance or the raw `ParseTree` stored in its `genes` attribute.
4. **Evaluate gameplay** – Call `play_GPac(pac_controller, **config['game'])`. The fitness
   function evaluates every successor state returned by GPac using your tree and selects the
   action with the highest score. The returned score/log pair can be fed directly to the
   visualization helpers later in the notebook.

## Tips
- Because terminal values are cached per state, you can experiment with deeper trees without a
  major performance penalty.
- The safe division logic in `/` returns the numerator when the denominator is zero, keeping
  evaluations stable even when states produce degenerate input.
- If you prefer deterministic behaviour from the `RAND` primitive while debugging, pass a
  seeded `random.Random` instance to `ParseTree.evaluate(..., rng=my_rng)`.
