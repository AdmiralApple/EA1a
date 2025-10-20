# Using the Grow Method in Assignment 2a

The newly implemented `TreeGenotype.generate_grow_tree` helper mirrors the structure of the
`generate_full_tree` method from the same module. It accepts the same keyword arguments that
are supplied via the notebook configuration loader:

```python
TreeGenotype.generate_grow_tree(
    depth_limit=config['pac_init']['depth_limit'],
    terminals=config['pac_init']['terminals'],
    nonterminals=config['pac_init']['nonterminals']
)
```

To complete the "The Grow Method for Tree Initialization" exercise in `2a_notebook.ipynb`,
update the provided loop so that it calls the helper for each depth you want to explore. For
example:

```python
for depth in [1, 2, 3, 4, 5]:
    tree = TreeGenotype.generate_grow_tree(
        depth_limit=depth,
        terminals=config['pac_init']['terminals'],
        nonterminals=config['pac_init']['nonterminals']
    )
    print(tree.root)
```

This will build trees whose branches are allowed to terminate early whenever a terminal is
selected, matching the algorithm described in the notebook section. The generated `ParseTree`
instances can then be inspected, serialized, or used in later exercises without additional
changes.
