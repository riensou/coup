from player import *
from game import *

player_1 = Player('Player 1', RANDOM_FUNCS)
player_2 = Player('Player 2', RANDOM_FUNCS)
player_3 = Player('Player 3', RANDOM_FUNCS)
player_4 = Player('Player 4', USER_FUNCS)

players = [player_1, player_2, player_3, player_4]

game = Game(players)

while len(game.game_state['players']) > 1:
    game.simulate_turn()
print("Winner:", game.game_state['players'][0].name)