# Tree Execution in the GPac Notebook

The tree execution utilities live in `tree_genotype.ParseTree` and are surfaced through `TreeGenotype.evaluate`. With these helpers in place, you can complete the **"Tree Execution"** section of `2a_notebook` without modifying the notebook itself.

## Preparing a Controller
1. Build or deserialize a `TreeGenotype` so that the `genes` attribute references a parse tree.
2. Use `TreeGenotype.evaluate(state, active_player=player_id)` if you want to experiment in a REPL cell, or simply pass the genotype to `play_GPac` as the `pac_controller`. The fitness helper automatically calls the genotype's `evaluate` method for each successor state.

```python
from tree_genotype import TreeGenotype
from fitness import play_GPac

# Assume `config` has already been loaded from green_config.txt
controller = TreeGenotype()
controller.genes = TreeGenotype.generate_full_tree(
    depth_limit=2,
    terminals=config['pac_init']['terminals'],
    nonterminals=config['pac_init']['nonterminals']
)

score, log = play_GPac(controller, **config['game'])
```

## Using the Evaluation Helpers Manually
When testing individual primitives, evaluate the parse tree directly. The evaluator caches terminal results per state to avoid redundant lookups.

```python
state = game.get_observations(['hold'], 'm')[0]
value = controller.evaluate(state, active_player='m')
print(f"Tree score for staying put: {value}")
```

The evaluator expects each primitive described in the notebook:

* `G`, `P`, and `F` compute Manhattan distances using the helper in `fitness.py`.
* `W` counts adjacent walls, treating out-of-bounds squares as walls.
* `C` returns the node's constant, which is sampled at creation time.

## Integrating with the Notebook
With `TreeGenotype.evaluate` and the updated `play_GPac`, the "Tree Execution" cell only needs to construct a controller and assign it to `pac_controller`. The game loop invokes the controller for each candidate state, applies the greedy policy specified by `pac_policy`, and plays a full game.

Use the `pac_policy` keyword argument to toggle between maximising or minimising state values:

```python
score, log = play_GPac(controller, pac_policy='min', **config['game'])
```

This combination of helpers covers the tasks outlined in the notebook section while keeping the notebook file unchanged.
