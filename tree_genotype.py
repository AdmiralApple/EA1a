
# tree_genotype.py

import random
from copy import deepcopy
from math import isfinite
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

    def evaluate(self, state, *, player=None, rng=random, cache=None):
        """
        Evaluate the parse tree against the provided game state.

        Parameters
        ----------
        state: dict
            Observation dictionary returned by ``GPacGame.get_observations``.
        player: str | None
            Identifier for the agent whose action produced ``state``.
            When ``None`` the function attempts to infer the lone Pac-Man.
        rng: random.Random | module
            Random number generator used by the RAND primitive.
        cache: dict | None
            Optional cache allowing terminal primitive values to be reused
            during a single evaluation.
        """

        if self.root is None:
            raise ValueError("parse tree must have a root node before evaluation")

        if state is None:
            raise ValueError("state must be a mapping produced by GPac")

        players = state.get("players")
        if not players:
            raise ValueError("state does not contain player locations")

        if player is None:
            # Fall back to the first Pac-Man identifier if the caller did not
            # explicitly provide the acting agent.
            pac_players = [name for name in players if "m" in name]
            if len(pac_players) != 1:
                raise ValueError("unable to infer acting player from state")
            player = pac_players[0]

        terminal_cache = cache if cache is not None else {}

        return self._evaluate_node(self.root, state, player, terminal_cache, rng)

    # ------------------------------------------------------------------
    # Internal helpers for tree execution.
    # ------------------------------------------------------------------
    def _evaluate_node(self, node, state, player, cache, rng):
        if node.is_terminal:
            return self._evaluate_terminal(node, state, player, cache)

        if len(node.children) != 2:
            raise ValueError(
                f"nonterminal primitive '{node.primitive}' expected 2 children"
            )

        left_value = self._evaluate_node(node.children[0], state, player, cache, rng)
        right_value = self._evaluate_node(node.children[1], state, player, cache, rng)

        return self._apply_operator(node.primitive, left_value, right_value, rng)

    def _evaluate_terminal(self, node, state, player, cache):
        primitive = node.primitive
        if primitive == "C":
            # Constant nodes embed their sampled value directly in the node.
            if node.value is None:
                raise ValueError("constant node is missing an assigned value")
            return float(node.value)

        key = (primitive, player)
        if key in cache:
            return cache[key]

        walls = state.get("walls")
        pills = state.get("pills", frozenset())
        fruit = state.get("fruit")
        players = state["players"]
        location = players.get(player)
        if location is None:
            raise KeyError(f"player '{player}' not present in state")

        width = len(walls)
        height = len(walls[0]) if width else 0

        if primitive == "G":
            if "m" in player:
                targets = [pos for name, pos in players.items() if "m" not in name]
            else:
                targets = [pos for name, pos in players.items() if "m" in name]
            cache[key] = self._nearest_distance(location, targets, width + height)
        elif primitive == "P":
            cache[key] = self._nearest_distance(location, pills, 0.0)
        elif primitive == "F":
            if fruit is None:
                cache[key] = float(width + height)
            else:
                cache[key] = float(manhattan(location, fruit))
        elif primitive == "W":
            cache[key] = float(
                self._adjacent_wall_count(location, walls, width, height)
            )
        else:
            raise ValueError(f"unsupported terminal primitive '{primitive}'")

        return cache[key]

    def _apply_operator(self, primitive, left_value, right_value, rng):
        if primitive == "+":
            return left_value + right_value
        if primitive == "-":
            return left_value - right_value
        if primitive == "*":
            return left_value * right_value
        if primitive == "/":
            if right_value == 0:
                # Gracefully degrade division by returning the numerator.
                return left_value
            return left_value / right_value
        if primitive == "RAND":
            low = min(left_value, right_value)
            high = max(left_value, right_value)
            if not isfinite(low) or not isfinite(high):
                # Degenerate ranges should not cause the generator to fail.
                return low if low == high else 0.0
            return rng.uniform(low, high)
        raise ValueError(f"unsupported nonterminal primitive '{primitive}'")

    @staticmethod
    def _nearest_distance(origin, targets, default_value):
        # Compute the distance to the closest element in ``targets``.
        if not targets:
            return float(default_value)
        return float(min(manhattan(origin, target) for target in targets))

    @staticmethod
    def _adjacent_wall_count(location, walls, width, height):
        # Count walls in the four cardinal directions, treating borders as walls.
        x, y = location
        deltas = ((1, 0), (-1, 0), (0, 1), (0, -1))
        count = 0
        for dx, dy in deltas:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                count += 1
                continue
            if walls[nx][ny]:
                count += 1
        return count

class TreeGenotype():
    def __init__(self):
        self.fitness = None
        self.genes = None

    def evaluate(self, state, *, player=None, rng=random, cache=None):
        """
        Delegate evaluation to the underlying parse tree.

        This thin wrapper keeps the genotype interface convenient when the
        notebook or fitness function stores the parse tree in ``genes``.
        """

        if self.genes is None:
            raise ValueError("genes must be initialized before evaluation")
        return self.genes.evaluate(state, player=player, rng=rng, cache=cache)

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
        """
        Return a depth first serialization of the active tree """

        if self.genes is None or self.genes.root is None:
            raise ValueError("genes must not be none")

        lines = []

        def visit(node, depth):
            """
            append the formatted for node and visit children """

            if node is None:
                return

            if node.primitive == "C" and node.value is not None:
                primitive_repr = str(float(node.value))
            else:
                primitive_repr = str(node.primitive)

            lines.append(f"{'|' * depth}{primitive_repr}")

            for child in node.children:
                visit(child, depth + 1)

        visit(self.genes.root, 0)

        return "\n".join(lines)


    def deserialize(self, serialization):
        """
        rebuild the parse tree represented by serialization """

        if not serialization:
            raise ValueError("serialization must be a non-empty string")

        #break the serialized form into meaningful lines
        raw_lines = [line for line in serialization.splitlines() if line]
        if not raw_lines:
            raise ValueError("serialization did not contain any nodes")

        def create_node(token):
            """
            creates a TreeNode from the serialized token """

            try:
                value = float(token)
            except ValueError:
                return TreeNode(token)
            return TreeNode("C", value=value)

        #process the first line separately so we can seed the stack with the root
        root_depth = len(raw_lines[0]) - len(raw_lines[0].lstrip('|'))
        if root_depth != 0:
            raise ValueError("root node must have depth 0")
        root_token = raw_lines[0][root_depth:]
        root = create_node(root_token)

        #the stack stores the most recent node encountered at each depth so that newly parsed nodes can be attached to their parent
        depth_stack = [root]

        for line in raw_lines[1:]:
            depth = len(line) - len(line.lstrip('|'))
            token = line[depth:]
            node = create_node(token)

            if depth == 0:
                raise ValueError("serialization contains multiple root nodes")

            #ensure the stack only contains nodes on the path to the current depth
            if depth > len(depth_stack):
                raise ValueError("invalid serialization: depth jumps more than one level")
            depth_stack = depth_stack[:depth]

            parent = depth_stack[depth - 1]
            parent.children.append(node)
            depth_stack.append(node)

        self.genes = ParseTree(root)


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

