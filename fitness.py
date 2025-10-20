
# fitness.py

import gpac
import random
from functools import cache
from math import inf, isnan


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Fitness function that plays a game using the provided pac_controller
# with optional ghost controller and game map specifications.
# Returns Pac-Man score from a full game as well as the game log.
def _resolve_controller(controller):
    """Return an evaluation callable for the provided controller."""

    if controller is None:
        return None

    evaluator = getattr(controller, "evaluate", None)
    if callable(evaluator):
        return evaluator

    genes = getattr(controller, "genes", None)
    evaluator = getattr(genes, "evaluate", None)
    if callable(evaluator):
        return evaluator

    raise TypeError("controller must expose an evaluate(state, active_player=...) method")


def _select_action_index(evaluator, policy, player, states):
    """Score successor states and return the greedy action index."""

    if not states:
        raise ValueError("states must contain at least one observation")

    best_index = 0
    best_score = -inf if policy == "max" else inf

    for index, state in enumerate(states):
        raw_score = evaluator(state, active_player=player)
        score = float(raw_score)
        if isnan(score):
            score = -inf if policy == "max" else inf

        if (policy == "max" and score > best_score) or (policy == "min" and score < best_score):
            best_score = score
            best_index = index

    return best_index


def play_GPac(pac_controller, ghost_controller=None, game_map=None, score_vector=False, **kwargs):
    pac_policy = kwargs.pop("pac_policy", "max").lower()
    ghost_policy = kwargs.pop("ghost_policy", "max").lower()
    if pac_policy not in {"max", "min"}:
        raise ValueError("pac_policy must be either 'max' or 'min'")
    if ghost_policy not in {"max", "min"}:
        raise ValueError("ghost_policy must be either 'max' or 'min'")

    game_map = parse_map(game_map)
    game = gpac.GPacGame(game_map, **kwargs)

    pac_evaluator = _resolve_controller(pac_controller)
    ghost_evaluator = _resolve_controller(ghost_controller)

    # Game loop, representing one turn.
    while not game.gameover:
        # Evaluate moves for each player.
        for player in game.players:
            actions = game.get_actions(player)
            s_primes = game.get_observations(actions, player)
            selected_action_idx = None

            # Select Pac-Man action(s) using provided strategy.
            if 'm' in player:
                if pac_controller is None:
                    # Random Pac-Man controller.
                    selected_action_idx = random.randrange(len(actions))

                else:
                    selected_action_idx = _select_action_index(pac_evaluator, pac_policy, player, s_primes)

            # Select Ghost action(s) using provided strategy.
            else:
                if ghost_controller is None:
                    # Random Ghost controller.
                    selected_action_idx = random.randrange(len(actions))

                else:
                    selected_action_idx = _select_action_index(ghost_evaluator, ghost_policy, player, s_primes)

            game.register_action(actions[selected_action_idx], player)
        game.step()
    if not score_vector:
        return game.score, game.log
    return game.score, game.log, game.score_vector


# Function for parsing map contents.
# Note it is cached, so modifying a file requires a kernel restart.
@cache
def parse_map(path_or_contents):
    if not path_or_contents:
        # Default generic game map, with a cross-shaped path.
        size = 21
        game_map = [[True for __ in range(size)] for _ in range(size)]
        for i in range(size):
            game_map[0][i] = False
            game_map[i][0] = False
            game_map[size//2][i] = False
            game_map[i][size//2] = False
            game_map[-1][i] = False
            game_map[i][-1] = False
        return tuple(tuple(y for y in x) for x in game_map)

    if isinstance(path_or_contents, str):
        if '\n' not in path_or_contents:
            # Parse game map from file path.
            with open(path_or_contents, 'r') as f:
                lines = f.readlines()
        else:
            # Parse game map from a single string.
            lines = path_or_contents.split('\n')
    elif isinstance(path_or_contents, list) and isinstance(path_or_contents[0], str):
        # Parse game map from a list of strings.
        lines = path_or_contents[:]
    else:
        # Assume the game map has already been parsed.
        return path_or_contents

    for line in lines:
        line.strip('\n')
    firstline = lines[0].split(' ')
    width, height = int(firstline[0]), int(firstline[1])
    game_map = [[False for y in range(height)] for x in range(width)]
    y = -1
    for line in lines[1:]:
        for x, char in enumerate(line):
            if char == '#':
                game_map[x][y] = True
        y -= 1
    return tuple(tuple(y for y in x) for x in game_map)
