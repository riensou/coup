from player import *
from game import *

player_1 = Player('Player 1', RANDOM_FUNCS)
player_2 = Player('Player 2', TRUTH_FUNCS)
player_3 = Player('Player 3', TRUTH_FUNCS)
player_4 = Player('Player 4', TRUTH_FUNCS)

players = [player_1, player_2, player_3, player_4]

game = Game(players, debug=True)

while len(game.game_state['players']) > 1:
    game.simulate_turn()
print("Winner:", game.game_state['players'][0].name)