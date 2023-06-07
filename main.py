from player import *
from rl_player import *
from game import *

player_1 = Player('Player 1', GREEDY_FUNCS)
player_2 = Player('Player 2', GREEDY_FUNCS)
player_3 = Player('Player 3', TRUTH_FUNCS)
player_4 = Player('Player 4', TRUTH_FUNCS)

players = [player_1, player_2, player_3, player_4]

game = Game(players)

while len(game.game_state['players']) > 1:
    game.simulate_turn()
    if game.game_state['current_player'].name == 'Player 1':
        print(state_to_input(game.game_state, game.history, 'Player 1'))
print("Winner:", game.game_state['players'][0].name)

# TODO: when player dies, set their coins to 0