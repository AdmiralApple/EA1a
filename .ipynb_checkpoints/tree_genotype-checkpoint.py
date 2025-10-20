
# tree_genotype.py

import random
from copy import deepcopy
from fitness import manhattan


class TreeNode:
    """
    Simple node for parse trees"""

    CONSTANT_RANGE = (-5.0, 5.0)

    def __init__(self, primitive, children=None, value=None):
        #store the primitives identifying string
        self.primitive = primitive
        #children default to an empty
        self.children = list(children) if children else []
        self.value = value

    @property
    def is_terminal(self):
        """
        Return True when the node has no children """

        return len(self.children) == 0

    @classmethod
    def make_terminal(cls, primitive, *, rng=random):
        """
        Factory helper that builds terminal nodes """

        if primitive == "C":
            value = rng.uniform(*cls.CONSTANT_RANGE)
            return cls(primitive, value=value)
        return cls(primitive)

    def __repr__(self):
        details = f" value={self.value}" if self.value is not None else ""
        return f"TreeNode({self.primitive}{details}, children={len(self.children)})"


class ParseTree:
    """
    Container for the root node of a parse tree """

    def __init__(self, root):
        self.root = root

    def max_depth(self):
        """
        Compute the maximum depth for validation """

        def _depth(node):
            if not node.children:
                return 0
            return 1 + max(_depth(child) for child in node.children)

        return _depth(self.root)

class TreeGenotype():
    def __init__(self):
        self.fitness = None
        self.genes = None

    @staticmethod
    def generate_full_tree(depth_limit, *, terminals, nonterminals, rng=random):
        """
        Build a full tree where every branch reaches depth_limit """

        if depth_limit < 0:
            raise ValueError("depth_limit is negative")

        term_choices = tuple(terminals)
        nonterm_choices = tuple(nonterminals)

        if not term_choices:
            raise ValueError("need at least one terminal primitive")
        if depth_limit > 0 and not nonterm_choices:
            raise ValueError("nonterminal primitives are required for depth > 0")

        def build(depth):
            if depth == depth_limit:
                primitive = rng.choice(term_choices)
                return TreeNode.make_terminal(primitive, rng=rng)

            primitive = rng.choice(nonterm_choices)
            #GPac primitives are all binary, so construct two children.
            left_child = build(depth + 1)
            right_child = build(depth + 1)
            return TreeNode(primitive, children=[left_child, right_child])

        return ParseTree(build(0))


    @staticmethod
    def generate_grow_tree(depth_limit, *, terminals, nonterminals, rng=random):
        """
        Build a tree using the grow method"""

        if depth_limit < 0:
            raise ValueError("depth_limit is negative")

        term_choices = tuple(terminals)
        nonterm_choices = tuple(nonterminals)

        if not term_choices:
            raise ValueError("need at least one terminal primitive")

        def build(depth):
            #choose from both primitive sets until the depth limit is hit
            at_limit = depth == depth_limit
            choices = term_choices if at_limit else term_choices + nonterm_choices
            if not choices:
                raise ValueError("no primitives available to build the tree")

            primitive = rng.choice(choices)

            #only expand children when a nonterminal was picked before the limit
            if primitive in nonterm_choices and not at_limit:
                left_child = build(depth + 1)
                right_child = build(depth + 1)
                return TreeNode(primitive, children=[left_child, right_child])

            return TreeNode.make_terminal(primitive, rng=rng)

        return ParseTree(build(0))


    @classmethod
    def initialization(cls, mu, depth_limit, **kwargs):
        population = [cls() for _ in range(mu)]

        #pull out primitive sets
        terminals = kwargs.get("terminals")
        nonterminals = kwargs.get("nonterminals")
        rng = kwargs.get("rng", random)

        if not terminals:
            raise ValueError("terminals must contain at least one primitive")
        if depth_limit < 1:
            raise ValueError("depth_limit must be a positive integer")

        #determine how many individuals should come from each strategy
        num_full = mu // 2

        for index, individual in enumerate(population):
            max_depth = rng.randint(1, depth_limit)
            use_full = index < num_full

            if use_full:
                tree = cls.generate_full_tree(
                    max_depth,
                    terminals=terminals,
                    nonterminals=nonterminals,
                    rng=rng,
                )
            else:
                tree = cls.generate_grow_tree(
                    max_depth,
                    terminals=terminals,
                    nonterminals=nonterminals,
                    rng=rng,
                )

            
            individual.genes = tree

        return population


    def serialize(self):
        # 2a TODO: Return a string representing self.genes in the required format.
        return 'Unimplemented'


    def deserialize(self, serialization):
        # 2a TODO: Complete the below code to recreate self.genes from serialization,
        #          which is a string generated by your serialize method.
        #          We have provided logic for tree traversal to help you get started,
        #          but you need to flesh out this function and make the genes yourself.

        lines = serialization.split('\n')

        # TODO: Create the root node yourself here based on lines[0]
        root = None

        parent_stack = [(root, 0)]
        for line in lines[1:]:
            if not line:
                continue
            my_depth = line.count('|')
            my_primitive = line.strip('|')
            parent, parent_depth = parent_stack.pop()
            right_child = False
            while parent_stack and parent_depth >= my_depth:
                parent, parent_depth = parent_stack.pop()
                right_child = True

            # TODO: Create a node using the above variables as appropriate.
            node = None

            parent_stack.extend([(parent, parent_depth), \
                                 (node, my_depth)])

        # TODO: Use the data structure you've created to assign self.genes.
        self.genes = None


    def recombine(self, mate, depth_limit, **kwargs):
        child = self.__class__()

        # 2b TODO: Recombine genes of mate and genes of self to
        #          populate child's genes member variable.
        #          We recommend using deepcopy, but also recommend
        #          that you deepcopy the minimal amount possible.

        return child


    def mutate(self, depth_limit, **kwargs):
        mutant = self.__class__()
        mutant.genes = deepcopy(self.genes)

        # 2b TODO: Mutate mutant.genes to produce a modified tree.

        return mutant

