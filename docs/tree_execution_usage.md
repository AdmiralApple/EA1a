# Tree Execution in the 2a Notebook

With the tree execution helpers implemented in `tree_genotype.py` and the
`play_GPac` integration inside `fitness.py`, the "Tree Execution" section of
`2a_notebook.ipynb` can be exercised without modifying the notebook itself. This
note summarises how to wire everything together when working through that
section.

1. **Instantiate or load a parse tree**
   * Use `TreeGenotype.generate_full_tree(...)` or `TreeGenotype.generate_grow_tree(...)`
     to build a tree that obeys your configuration's terminal and nonterminal
     sets. Assign the resulting `ParseTree` object to a `TreeGenotype` instance's
     `genes` member or keep the `ParseTree` directly.
2. **Score observations**
   * The new `ParseTree.evaluate(state, actor=...)` method accepts the state
     dictionaries returned by `GPacGame.get_observations`. Make sure to include
     an `"active_player"` key (e.g., `state['active_player'] = 'm'`) before
     scoring so the evaluator knows which Pac-Man to treat as the focal agent.
   * Terminal primitives are computed exactly as prescribed in the notebook:
     `G`, `P`, and `F` use Manhattan distance without considering walls, `W`
     counts the cardinally adjacent walls, and `C` returns the node's constant.
     Repeated sensor lookups are cached automatically while evaluating a single
     state.
3. **Select actions inside the notebook cell**
   * Pass your `TreeGenotype` (or its `.genes`) to `play_GPac`. The
     implementation now resolves the backing tree and evaluates each candidate
     state in `s_primes`, selecting the action with the highest score.
   * Because the tree execution lives in the Python modules, re-run the notebook
     cell that imports `fitness` and `tree_genotype` after making changes so the
     updated logic is available to subsequent cells.

Following these steps allows the notebook cell that defines `pac_controller` to
run end-to-end: the controller evaluates each observation, caches sensor
calculations per state, and `play_GPac` greedily chooses the action whose score
is maximal according to the tree.
