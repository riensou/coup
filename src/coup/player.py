from coup.utils import *

import numpy as np
import random

class Player():
    def __init__(self, name, funcs):
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
    

# -- Random Ronald -- #
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

# -- Truth Teller Tim -- #
def truth_decision(game_state, history, name):
    player_cards = game_state['player_cards'][name]
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    truth_actions = [action for action in possible_actions if action[2] not in ACTION_SENDER.keys() or ACTION_SENDER[action[2]] in player_cards]
    return random.choice(truth_actions)

def truth_block(action, game_state, history, name, action_is_block=False):
    player_cards = game_state['player_cards'][name]
    if action_is_block:
        return (name, False, None)
    else:
        if ROLE_BLOCKABLE[action[2]] and not did_block_1_lie(action[2], player_cards):
            return (name, True, False)
        else:
            return (name, False, None)

TRUTH_FUNCS = {'decision_fn': truth_decision, 'block_fn': truth_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

# -- Greedy Garrett -- #

def greedy_decision(game_state, history, name):
    player_cards = game_state['player_cards'][name]
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    greedy_actions = [action for action in possible_actions if action[2] == 'Income' or action[2] == 'Foreign Aid' or action[2] == 'Tax' or action[2] == 'Assassinate' or action[2] == 'Coup']
    return random.choice(greedy_actions)

def greedy_block(action, game_state, history, name, action_is_block=False):
    player_cards = game_state['player_cards'][name]
    if action_is_block:
        return (name, False, None)
    else:
        if action[1] == name:
            if ROLE_BLOCKABLE[action[2]] and not did_block_1_lie(action[2], player_cards):
                return (name, True, False)
            elif LIE_BLOCKABLE[action[2]]:
                return (name, True, True)
            elif ROLE_BLOCKABLE[action[2]]:
                return (name, True, False)
        return (name, False, None)

GREEDY_FUNCS = {'decision_fn': greedy_decision, 'block_fn': greedy_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

# -- Income Isabel -- #

def income_decision(game_state, history, name):
    player_cards = game_state['player_cards'][name]
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    income_actions = [action for action in possible_actions if action[2] == 'Income' or action[2] == 'Assassinate']
    if income_actions:
        return random.choice(income_actions)
    else:
        return random.choice(possible_actions)
    
def income_block(action, game_state, history, name, action_is_block=False):
    player_cards = game_state['player_cards'][name]
    if action_is_block:
        return (name, False, None)
    else:
        if action[1] == name and ROLE_BLOCKABLE[action[2]] and not did_block_1_lie(action[2], player_cards):
            return (name, True, False)
    return (name, False, None)

INCOME_FUNCS = {'decision_fn': income_decision, 'block_fn': income_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

# -- User Ulysses -- #

def user_decision(game_state, history, name):
    print("It's your turn, {}".format(name))
    print("Available actions:")
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    for i, action in enumerate(possible_actions):
        print("{}. {}".format(i + 1, action))
    
    while True:
        choice = input("Choose an action (enter the number): ")
        try:
            index = int(choice) - 1
            if index >= 0 and index < len(possible_actions):
                return possible_actions[index]
        except ValueError:
            pass

        print("Invalid choice. Please try again.")

def user_block(action, game_state, history, name, action_is_block=False): 
    if not action_is_block:
        print("Do you want to block? (Y/N): ")
        while True:
            choice = input().lower()
            if choice == 'y':
                print("Do you block by calling bluff or counter action (0/1): ")
                while True:
                    choice = input()
                    if choice == '0':
                        return (name, True, True)
                    elif choice == '1':
                        return (name, True, False)
                    else:
                        print("Invalid choice. Please enter 0 or 1.")
            elif choice == 'n':
                return (name, False, None)
            else:
                print("Invalid choice. Please enter Y or N.")
    elif not action[2]:
        print("Do you want to call their bluff? (Y/N)")
        while True:
            choice = input().lower()
            if choice == 'y':
                return (name, True, True)
            elif choice == 'n':
                return (name, False, None)
            else:
                print("Invalid choice. Please enter Y or N.")
    else:     
        return (name, False, None)

def user_dispose(game_state, history, name):
    print("Your cards: {}".format(game_state['player_cards'][name]))
    print("Choose a card to dispose (enter the index): ")
    while True:
        choice = input()
        try:
            index = int(choice)
            if index >= 0 and index < len(game_state['player_cards'][name]):
                return index
        except ValueError:
            pass

        print("Invalid choice. Please try again.")

def user_keep(cards, game_state, history, name):
    print("Your cards: {}".format(cards))
    print("Choose the indices of the cards to keep (enter as comma-separated values): ")
    while True:
        choice = input()
        indices = [int(idx) for idx in choice.split(",") if idx.strip().isdigit()]
        if len(indices) == len(game_state['player_cards'][name]) and all(idx >= 0 and idx < len(cards) for idx in indices):
            return indices

        print("Invalid choice. Please try again.")

USER_FUNCS = {'decision_fn': user_decision, 'block_fn': user_block, 'dispose_fn': user_dispose, 'keep_fn': user_keep}

# -- Self Player Samuel -- #

class SelfPlayer(Player):
    def __init__(self, name, funcs, model, idx, n, k):
        super().__init__(name, funcs)
        
        self.model = model
        self.idx = idx
        self.n = n
        self.k = k
        
    def decision(self, game_state, history):
        return self.decision_fn(game_state, history, model=self.model, idx=self.idx, n=self.n, k=self.k)
    
    def block(self, action, game_state, history, action_is_block=False):
        return self.block_fn(action, game_state, history, model=self.model, idx=self.idx, n=self.n, k=self.k, action_is_block=action_is_block)
    
    def dispose(self, game_state, history):
        return self.dispose_fn(game_state, history, model=self.model, idx=self.idx, n=self.n, k=self.k)
    
    def keep(self, cards, game_state, history):
        return self.keep_fn(cards, game_state, history, model=self.model, idx=self.idx, n=self.n, k=self.k)

def self_decision(game_state, history, model, idx, n=4, k=10):
    observation = np.concatenate((encode_gamestate(game_state, idx, n), encode_history(game_state, history, idx, n, k)))
    a, _ = model.predict(observation)
    a = a[0:1+3*n]
    player_names = list(game_state['player_deaths'].keys())
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    list_of_players = list([p_name for p_name in game_state['player_deaths'].keys() if p_name != player_names[idx]])
    idx_to_player = {i : list_of_players[i] for i in range(len(list_of_players))}
    idx_to_type = {0: 'Income', 1: 'Foreign Aid', 2: 'Tax', 3: 'Exchange'}
    for i in range(4, 3 + n):
        idx_to_type[i] = 'Steal'
    for i in range(3 + n, 2 + 2 * n):
        idx_to_type[i] = 'Assassinate'
    for i in range(2 + 2 * n, 1 + 3 * n):
        idx_to_type[i] = 'Coup'
    action = None
    while action not in possible_actions:
        i = np.argmax(a)
        a[i] = -1 * float('inf')
        type = idx_to_type[i]
        if i > 3:
            reciever = idx_to_player[(i - 4) % (n - 1)]
        else:
            reciever = player_names[idx]
        action = (player_names[idx], reciever, type)
    return action

def self_block(action, game_state, history, model, idx, n=4, k=10, action_is_block=False):
    observation = np.concatenate((encode_gamestate(game_state, idx, n), encode_history(game_state, history, idx, n, k)))
    a, _ = model.predict(observation)
    player_names = list(game_state['player_deaths'].keys())
    if not action_is_block:
        a = a[1+3*n:4+3*n]
        possible_blocks = generate_all_blocks(player_names[idx], action)
        block1 = None
        while block1 not in possible_blocks:
            i = np.argmax(a)
            a[i] = -1 * float('inf')
            if i == 0: # accept
                block1 = (player_names[idx], False, None)
            if i == 1: # dispute
                block1 = (player_names[idx], True, True)
            if i == 2: # block
                block1 = (player_names[idx], True, False)
        return block1
    else:
        a = a[4+3*n:6+3*n]
        if action[2]:
            action = (action[0], player_names[idx], 'Lie_Block')
        else:
            action = (action[0], player_names[idx], 'Role_Block')
        possible_blocks = generate_all_blocks(player_names[idx], action)
        block2 = None
        while block2 not in possible_blocks:
            i = np.argmax(a)
            a[i] = -1 * float('inf')
            if i == 0: # accept
                block2 = (player_names[idx], False, None)
            if i == 1: # dispute
                block2 = (player_names[idx], True, True)  
        return block2

def self_dispose(game_state, history, model, idx, n=4, k=10):
    observation = np.concatenate((encode_gamestate(game_state, idx, n), encode_history(game_state, history, idx, n, k)))
    a, _ = model.predict(observation)
    a = a[6+3*n:8+3*n]
    i = np.argmax(a)
    player_names = list(game_state['player_deaths'].keys())
    if len(game_state['player_deaths'][player_names[idx]]) > 0: i = 0
    return i

def self_keep(cards, game_state, history, model, idx, n=4, k=10):
    observation = np.concatenate((encode_gamestate(game_state, idx, n), encode_history(game_state, history, idx, n, k)))
    a, _ = model.predict(observation)
    a = a[8+3*n:14+3*n]
    player_names = list(game_state['player_deaths'].keys())
    if len(game_state['player_cards'][player_names[idx]]) < 4:
        i = np.argmax(a[0:3])
        idx_to_keep = {0: [0], 1: [1], 2: [2]}
        return idx_to_keep[i]
    else:
        i = np.argmax(a)
        idx_to_keep = {0: [0, 1],
                        1: [0, 2],
                        2: [0, 3],
                        3: [1, 2],
                        4: [1, 3],
                        5: [2, 3]}
        return idx_to_keep[i]

SELF_FUNCS = {'decision_fn': self_decision, 'block_fn': self_block, 'dispose_fn': self_dispose, 'keep_fn': self_keep}

def encode_gamestate(game_state, idx, n):
    players, deck, player_cards, player_deaths, player_coins, current_player = game_state.values()
    player_names = list(player_deaths.keys())
    name = player_names[idx]
    our_cards = player_cards[name]

    encoding = np.zeros((20 + 12 * n,))

    # fill [0 : 10] with information about our_cards
    if len(player_cards[name]) > 0:
        encoding[ROLE_TO_I[our_cards[0]]] = 1
    if len(player_cards[name]) > 1:
        encoding[ROLE_TO_I[our_cards[1]] + 5] = 1
    # fill [10 : 20] with more information about our_cards (if during an exhchange)
    if len(player_cards[name]) > 2:
        encoding[ROLE_TO_I[our_cards[2]] + 10] = 1
    if len(player_cards[name]) > 3:
        encoding[ROLE_TO_I[our_cards[3]] + 15] = 1
    # fill [20 : 20+n] with information about player_coins
    for i in range(n):
        player_name = player_names[i]
        encoding[20 + i] = player_coins[player_name] / 12
    # fill [20+n : 20+11n] entries with information about player_deaths
    for i in range(n):
        player_name = player_names[i]
        if len(player_deaths[player_name]) > 0:
            encoding[20 + n + 10 * i + ROLE_TO_I[player_deaths[player_name][0]]] = 1
        if len(player_deaths[player_name]) > 1:
            encoding[20 + n + 10 * i + 5 + ROLE_TO_I[player_deaths[player_name][1]]] = 1
    # fill [20+11n : 20+12n] with information about which player you are
    encoding[20 + 11 * n + idx] = 1

    return encoding

def encode_history(game_state, history, idx, n, k):
    encoding = np.zeros(((6 * n + 35) * k,))
    encoded_turns, event_encoding = 0, np.zeros((6 * n + 35,))
    for event in reversed(history):
        if encoded_turns == k:
            return encoding
        if event[0] == 'a':
            event_encoding[0:4+4*n] = encode_action_event(event[1], game_state, n)
            encoding[(6*n+35)*encoded_turns:(6*n+35)*(encoded_turns+1)] = event_encoding
            encoded_turns += 1
            event_encoding = np.zeros((6 * n + 35,))
        elif event[0] == 'b1':
            event_encoding[4+4*n:7+5* n] = encode_block1_event(event[1], game_state, n)
        elif event[0] == 'b2':
            event_encoding[7+5*n:9+6*n] = encode_block2_event(event[1], game_state, n)
        elif event[0] == f"k{idx}":
            event_encoding[9+6*n:35+6*n] = encode_keep_event(event[1], game_state, n)
    return encoding

def encode_action_event(action, game_state, n):
    action_encoding = np.zeros((4*n+4,))
    sender, reciever, action_type = action
    player_names = list(game_state['player_deaths'].keys())

    # encode the sender
    action_encoding[player_names.index(sender)] = 1

    # encode the action_type along with reciever
    if action_type == 'Income':
        action_encoding[n] = 1
    elif action_type == 'Foreign Aid':
        action_encoding[n + 1] = 1
    elif action_type == 'Tax':
        action_encoding[n + 2] = 1
    elif action_type == 'Exchange':
        action_encoding[n + 3] = 1
    elif action_type == 'Steal':
        action_encoding[n + 4 + player_names.index(reciever)] = 1
    elif action_type == 'Assassinate':
        action_encoding[2 * n + 4 + player_names.index(reciever)] = 1 
    elif action_type == 'Coup':
        action_encoding[3 * n + 4 + player_names.index(reciever)] = 1
    else:
        exit(1)

    return action_encoding

def encode_block1_event(block1, game_state, n):
    block1_encoding = np.zeros((3+n,))
    sender, not_accept, refute_or_block = block1

    # encode accept / refute / block
    if not not_accept:
        block1_encoding[0] = 1
    elif refute_or_block:
        block1_encoding[1] = 1
    else:
        block1_encoding[2] = 1

    # encode the blocker
    if not_accept:
        player_names = list(game_state['player_deaths'].keys())
        block1_encoding[3 + player_names.index(sender)] = 1

    return block1_encoding

def encode_block2_event(block2, game_state, n):
    block2_encoding = np.zeros((2+n,))
    sender, not_accept, _ = block2

    # encode accept / refute
    if not not_accept:
        block2_encoding[0] = 1
    else:
        block2_encoding[1] = 1

    # encode the blocker
    if not_accept:
        player_names = list(game_state['player_deaths'].keys())
        block2_encoding[2 + player_names.index(sender)] = 1

    return block2_encoding

def encode_keep_event(keep, game_state, n):
    keep_encoding = np.zeros((26,))
    cards, card_idxs = keep

    # encode the cards
    card_to_encoding = {'Duke': 0, 
                        'Assassin': 1, 
                        'Captain': 2, 
                        'Ambassador': 3, 
                        'Contessa': 4}
    for i, card in enumerate(cards):
        keep_encoding[5*i+card_to_encoding[card]] = 1

    # encode the card_idxs
    idxs_to_encoding = {frozenset({0, 1}): 20,
                        frozenset({0, 2}): 21,
                        frozenset({0, 3}): 22,
                        frozenset({1, 2}): 23,
                        frozenset({1, 3}): 24,
                        frozenset({2, 3}): 25,
                        frozenset({0}): 20,
                        frozenset({1}): 21,
                        frozenset({2}): 22}
    keep_encoding[idxs_to_encoding[frozenset(card_idxs)]] = 1

    return keep_encoding