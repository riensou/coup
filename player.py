from utils import *

class Player():
    def __init__(self, name, influence, coins, cards, funcs):
        # basic player stats
        self.name = name
        self.influence = influence
        self.coins = coins
        self.cards = cards

        # player logic
        self.decide_fn = funcs['decision_fn']
        self.block_fn = funcs['block_fn']
        self.dispose_fn = funcs['dispose_fn']
        self.keep_fn = funcs['keep_fn']

    def decide(self, game_state, history):
        """
        Return an action of the form (sender, reciever, type) based on the game_state and history of the game. 
        
        sender = this player (always)
        reciever = any player
        type = Unblockable, Duke, Assassin, Captain, Ambassador, Contessa
        """
        return self.decide_fn(game_state, history)
    
    def block(self, action, game_state, history, action_is_block=False):
        """
        Return a block of the form (blocker, attempt_block, lie_or_counter) based on the action, game_state, and history of the game.
        
        blocker = this player (always)
        attempt_block = boolean value
        lie_or_counter = boolean value; True if calling the action a lie, False if claiming a counter role
        """
        return self.block_fn(action, game_state, history, action_is_block=action_is_block)
    
    def dispose(self, game_state, history):
        """
        Return the index of the card from self.cards to remove given the fact that a card must be removed based on the game_state and history of the game.
        """
        return self.dispose_fn(game_state, history)
    
    def keep(self, cards, game_state, history):
        """
        Return the indices of the cards to keep from cards based on the game_state and history of the game.
        """
        return self.keep_fn(cards, game_state, history)
    
def random_decision(game_state, history):
    pass

def random_block(action, game_state, history, action_is_block=False):
    pass

def random_dispose(game_state, history):
    pass

def random_keep(cards, game_state, history):
    pass
