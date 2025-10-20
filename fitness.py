
# fitness.py

import gpac
import random
from functools import cache
from math import inf


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Fitness function that plays a game using the provided pac_controller
# with optional ghost controller and game map specifications.
# Returns Pac-Man score from a full game as well as the game log.
def play_GPac(pac_controller, ghost_controller=None, game_map=None, score_vector=False, **kwargs):
    game_map = parse_map(game_map)
    game = gpac.GPacGame(game_map, **kwargs)

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
                    '''
                    ####################################
                    ###   YOUR 2a CODE STARTS HERE   ###
                    ####################################
                    '''
                    # 2a TODO: Score all of the states stored in s_primes by evaluating your tree.

                    # 2a TODO: Assign index of state with the best score to selected_action_idx.
                    selected_action_idx = None

                    # You may want to uncomment these print statements for debugging.
                    # print(selected_action_idx)
                    # print(actions)

                    '''
                    ####################################
                    ###    YOUR 2a CODE ENDS HERE    ###
                    ####################################
                    '''

            # Select Ghost action(s) using provided strategy.
            else:
                if ghost_controller is None:
                    # Random Ghost controller.
                    selected_action_idx = random.randrange(len(actions))

                else:
                    '''
                    ####################################
                    ###   YOUR 2c CODE STARTS HERE   ###
                    ####################################
                    '''
                    # 2c TODO: Score all of the states stored in s_primes by evaluating your tree

                    # 2c TODO: Assign index of state with the best score to selected_action_idx.
                    selected_action_idx = None

                    # You may want to uncomment these print statements for debugging.
                    # print(selected_action_idx)
                    # print(actions)
                    
                    '''
                    ####################################
                    ###    YOUR 2c CODE ENDS HERE    ###
                    ####################################
                    '''

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
