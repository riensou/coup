from utils import *
import random

class Game():
    def __init__(self, players):
        self.game_state = self.initial_gamestate(players)
        self.history = []

    def initial_gamestate(self, players):
        assert len(players) < 6

        deck = ROLES * 3
        random.shuffle(deck)

        player_cards = {}
        player_deaths = {}
        player_coins = {}
        for player in players:
            player_cards[player.name] = [deck.pop(), deck.pop()]
            player_deaths[player.name] = []
            player_coins[player.name] = 2

        current_player = players[0]

        game_state = {'players': players,
                      'deck': deck,
                      'player_cards': player_cards,
                      'player_deaths': player_deaths,
                      'player_coins': player_coins,
                      'current_player': current_player}
        
        return game_state
    
    def unpack_gamestate(self):
        return self.game_state['players'], self.game_state['deck'], self.game_state['player_cards'], self.game_state['player_deaths'], self.game_state['player_coins'], self.game_state['current_player']
    
    def simulate_turn(self):
        players, deck, player_cards, player_deaths, player_coins, current_player = self.unpack_gamestate()
        action, block_1, block_2 = None, None, None

        action = current_player.decision(self.game_state, self.history)

        for player in [p for p in players if p.name != current_player.name]:
            potential_block_1 = player.block(action, self.game_state, self.history)
            if potential_block_1[1]:
                block_1 = potential_block_1
                break

        for player in [p for p in players if p.name != block_1[0].name]:
            potential_block_2 = player.block(block_1, self.game_state, self.history, action_is_block=True)
            if potential_block_2[1]:
                block_2 = potential_block_2
                break

        turn = (action, block_1, block_2)
        self.history.append((self.game_state, turn))
        self.apply_game_logic(turn)

    def apply_game_logic(self, turn):
        action, block_1, block_2 = turn



    

    def take_action(self, action, cards=[], cards_idxs=[]):
        players, deck, player_cards, player_deaths, player_coins, current_player = self.unpack_gamestate()
        player1_name, player2_name, type = action

        if type == 'Income':
            income(player1_name, player_coins)
        elif type == 'Foreign Aid':
            foreign_aid(player1_name, player_coins)
        elif type == 'Tax':
            tax(player1_name, player_coins)
        elif type == 'Steal':
            steal(player1_name, player2_name, player_coins)
        elif type == 'Coup':
            assert len(cards_idxs) == 1
            coup(player1_name, player2_name, player_coins, player_cards, cards_idxs, player_deaths)
        elif type == 'Assassinate':
            assert len(cards_idxs) == 1
            assassinate(player1_name, player2_name, player_coins, player_cards, cards_idxs, player_deaths)
        elif type == 'Exchange':
            assert len(cards_idxs) == len(player_cards[player1_name])
            exchange(player1_name, player_cards, cards, player_cards, cards_idxs, deck)