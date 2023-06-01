from player import *
from game import *

random_player_1 = Player('Player 1', RANDOM_FUNCS)
random_player_2 = Player('Player 2', RANDOM_FUNCS)
random_player_3 = Player('Player 3', RANDOM_FUNCS)
random_player_4 = Player('Player 4', RANDOM_FUNCS)

players = [random_player_1, random_player_2, random_player_3, random_player_4]

game = Game(players)
game.simulate_turn()