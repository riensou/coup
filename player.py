from utils import *

import random

class Player():
    def __init__(self, name, funcs):
        # basic player stats
        self.name = name

        # player logic
        self.decision_fn = funcs['decision_fn']
        self.block_fn = funcs['block_fn']
        self.dispose_fn = funcs['dispose_fn']
        self.keep_fn = funcs['keep_fn']

    def decision(self, game_state, history):
        """
        Return an action of the form (sender, reciever, type) based on the game_state and history of the game. 
        
        sender = this player (always)
        reciever = any player
        type = Unblockable, Duke, Assassin, Captain, Ambassador, Contessa
        """
        return self.decision_fn(game_state, history, self.name)
    
    def block(self, action, game_state, history, action_is_block=False):
        """
        Return a block of the form (blocker, attempt_block, lie_or_counter) based on the action, game_state, and history of the game.
        
        blocker = this player (always)
        attempt_block = boolean value
        lie_or_counter = boolean value; True if calling the action a lie, False if claiming a counter role
        """
        return self.block_fn(action, game_state, history, self.name, action_is_block=action_is_block)
    
    def dispose(self, game_state, history):
        """
        Return the index of the card from self.cards to remove given the fact that a card must be removed based on the game_state and history of the game.
        """
        return self.dispose_fn(game_state, history, self.name)
    
    def keep(self, cards, game_state, history):
        """
        Return the indices of the cards to keep from cards based on the game_state and history of the game.
        """
        return self.keep_fn(cards, game_state, history, self.name)
    
def random_decision(game_state, history, name):
    return random.choice(generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards']))

def random_block(action, game_state, history, name, action_is_block=False):
    if action_is_block:
        if action[2]:
            action = (action[0], name, 'Lie_Block')
        else:
            action = (action[0], name, 'Role_Block')
    return random.choice(generate_all_blocks(name, action))

def random_dispose(game_state, history, name):
    return random.randint(0, len(game_state['player_cards'][name]) - 1)

def random_keep(cards, game_state, history, name):
    return random.sample(list(range(len(cards))), k=len(cards)-2)

RANDOM_FUNCS = {'decision_fn': random_decision, 'block_fn': random_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}
