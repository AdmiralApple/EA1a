# Using the Full Method for Tree Initialization

The `TreeGenotype.generate_full_tree` helper builds a parse tree where every branch reaches the requested depth. To mirror the expectations from the `2a_notebook`, import your configuration and pass the relevant primitives to the helper:

```python
config = read_config('configs/2a/green_config.txt', globals(), locals())
full_tree = TreeGenotype.generate_full_tree(
    depth_limit=3,
    terminals=config['pac_init']['terminals'],
    nonterminals=config['pac_init']['nonterminals']
)
```

The returned `ParseTree` instance exposes its root node through the `root` attribute, making it straightforward to integrate with later serialization or execution utilities. You can quickly verify the structure of a generated tree (and confirm that it satisfies the full method) by inspecting the maximum depth:

```python
print(full_tree.max_depth())  # should equal the depth_limit argument
```

Within the notebook section titled **"The Full Method for Tree Initialization"**, replace the `tree = None` placeholder with a call to `TreeGenotype.generate_full_tree(...)` using the loop's `depth` variable. Each iteration will now produce a new full tree matching the required depth, enabling you to visualize or test the initialization routine before implementing ramped half-and-half in subsequent sections.
