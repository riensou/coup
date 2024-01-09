import gymnasium as gym
import numpy as np
from gymnasium import spaces

from coup.utils import *
from coup.player import *
import random

class Coup(gym.Env):
    """
    Coup Environment that follows gym interface.

    n = number of players
    k = history length
    """

    def __init__(self, n=4, k=10):
        super().__init__()

        action_size = 4 + 3 * (n - 1) # 4 actions on oneself, 3 actions on other players
        block_1_size = 3 # accept, dispute, block
        block_2_size = 2 # accept, dispute
        dispose_size = 2 # remove card 1 or card 2
        keep_size = 6 # (4 choose 2)
        ac_dim = action_size + block_1_size + block_2_size + dispose_size + keep_size

        game_state_size = 20 + 11 * n # 20 (cards owned by agent, plus the exchange case) + n (number of coins per player) + 10n (dead cards per player)
        history_size = (35 + 6 * n) * k # use the last k turns as input
        ob_dim = game_state_size + history_size


        self.action_space = spaces.Box(low=0, high=1, shape=(ac_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(ob_dim,), dtype=np.float32)
        self.n, self.k = n, k

    def step(self, a):
        self._decode_action(a)

        self._run_phase_transition()

        terminated = (self.players[self.agent_idx] not in self.game_state['players']) or (len(self.game_state['players']) == 1)
        if not terminated:
            self._run_game_until_input()

        observation = self._observation()
        reward = self._reward()
        terminated = (self.players[self.agent_idx] not in self.game_state['players']) or (len(self.game_state['players']) == 1)
        truncated = self.i > 500
        info = {}
        if terminated or truncated:
            info["is_success"] = all(self.players[self.agent_idx].name == p for p in [p.name for p in self.game_state['players']])
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        options:
        - 'players': specifices the archetypes to be played against
        - 'agent_idx': specifies which player the agent is
        - 'reward_hyperparameters': specifies the values of features to be rewarded
            * coins
            * number of agent's cards
            * number of opponents' cards
            * value of winning

        example:
        - options = {'players': [Player(f"Player {i+1}", RANDOM_FUNCS) for i in range(self.n)], 'agent_idx': 0}
        """
        if options == None:
            self.players, self.agent_idx, self.reward_hyperparameters = [Player(f"Player {i+1}", random.choice([RANDOM_FUNCS, TRUTH_FUNCS, GREEDY_FUNCS, INCOME_FUNCS])) for i in range(self.n)], 0, [0.1, 1, -0.5, 100]
        else:
            self.players, self.agent_idx, self.reward_hyperparameters = options['players'], options['agent_idx'], options['reward_hyperparameters']

        self.game_state = self._initial_gamestate(self.players)
        self.history = []
        self.phase = "action"
        self.i = 0

        self._run_game_until_input()

        observation = self._observation()
        info = {}

        return observation, info
    
    def render(self):
        pass

    def close(self):
        pass


    def _initial_gamestate(self, players):
        assert len(players) <= 6

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

    def _run_game_until_input(self):
        """
        Runs the game until an input from the agent specified by self.agent_idx is required to continue. 

        Modifies self.game_state, self.history, self.phase

        self.phase is one of the following: "action", "block1", "block2", "dispose", or "keep"

        action: get action_action from current player
        block1: get action_block1 from all players besides current player (stop if dispute or block)
        block2: get action_block2 from all players besides block1er (stop if dispute)
        dispose: get action_dispose from player who lost a card
        keep: get action_keep from player who exchanged successfully
        """
        if (self.players[self.agent_idx] not in self.game_state['players']) or (len(self.game_state['players']) == 1):
            return

        players, deck, player_cards, player_deaths, player_coins, current_player = self.game_state.values()

        if self.phase == "action":
            self.current_action, self.current_block1, self.current_block2, self.current_dispose, self.current_keep = (None, None, None), (None, None, None), (None, None, None), [], None
            self.current_disposers, self.current_block1_queried, self.current_block2_queried = [], [], []
            self._debug()
            if self._decision_is_agent(current_player): 
                return 
            else:
                self.current_action = current_player.decision(self.game_state, self.history)
                self._action_phase_transition()
            
        elif self.phase == "block1":
            self._debug()
            for player in [p for p in players if p.name != current_player.name]:
                if player in self.current_block1_queried: continue
                self.current_block1_queried.append(player)
                if self._decision_is_agent(player):
                    return 
                else:
                    potential_block1 = player.block(self.current_action, self.game_state, self.history)
                    if potential_block1[1]:
                        self.current_block1 = potential_block1
                        break
            self._block1_phase_transition()
        elif self.phase == "block2":
            self._debug()
            for player in [p for p in players if p.name != self.current_block1[0]]:
                if player in self.current_block2_queried: continue
                self.current_block2_queried.append(player)
                if self._decision_is_agent(player):
                    return
                else:
                    potential_block2 = player.block(self.current_block1, self.game_state, self.history, action_is_block=True)
                    if potential_block2[1]:
                        self.current_block2 = potential_block2
                        break
            self._block2_phase_transition()
        elif self.phase == "dispose":
            if not self.current_disposers: self.current_disposers = self._determine_disposers()
            self._debug()
            agent_dispose = False
            for disposer in self.current_disposers:
                if self._decision_is_agent(disposer):
                    agent_dispose = True
                    continue
                else:
                    self.current_dispose.append((disposer, disposer.dispose(self.game_state, self.history)))
            if agent_dispose:
                return
            self._dispose_phase_transition()
        elif self.phase == "keep":
            self._debug()
            player_cards[current_player.name] += [deck.pop(), deck.pop()]
            if self._decision_is_agent(current_player):
                return
            else:
                self.current_keep = current_player.keep(player_cards[current_player.name], self.game_state, self.history)
                self._keep_phase_transition()
        else:
            exit(1)

        self.i += 1

        self._run_game_until_input()
            
    def _run_phase_transition(self):
        if self.phase == "action":
            self._action_phase_transition()
        elif self.phase == "block1":
            self._block1_phase_transition()
        elif self.phase == "block2":
            self._block2_phase_transition()
        elif self.phase == "dispose":
            self._dispose_phase_transition()
        elif self.phase == "keep":
            self._keep_phase_transition()
        else:
            exit(1)

    def _action_phase_transition(self):
        self.history.append(('a', self.current_action))
        if self.current_action[2] in ['Income']:
            self.phase = "action"
            self._simulate_turn()
        elif self.current_action[2] in ['Coup']:
            self.phase = "dispose"
        else:
            self.phase = "block1"

    def _block1_phase_transition(self):
        if len(self.current_block1_queried) < self.n - 1 : 
            self.phase = "block1"
        self.history.append(('b1', self.current_block1))
        if not self.current_block1[1]:
            if self.current_action[2] in ['Assassinate']:
                self.phase = "dispose"
            elif self.current_action[2] in ['Exchange']:
                self.phase = "keep"
            else:
                self.phase = "action"
                self._simulate_turn()
        else:
            if self.current_block1[2]:
                self.phase = "dispose"
            else:
                self.phase = "block2"

    def _block2_phase_transition(self):
        if len(self.current_block2_queried) < self.n - 1 : 
            self.phase = "block2"
        self.history.append(('b2', self.current_block2))
        if not self.current_block2[1]:
            self.phase = "action"
            self._simulate_turn()
        else:
            self.phase = "dispose"

    def _dispose_phase_transition(self):
        if self.current_action[2] in ['Exchange']:
            player_cards = self.game_state['player_cards']
            sender_cards = player_cards[self.current_action[0]]
            if not did_action_lie(self.current_action[2], sender_cards):
                self.phase = "keep"
            else:
                self.phase = "action"
                self._simulate_turn()
        else:
            self.phase = "action"
            self._simulate_turn()

    def _keep_phase_transition(self):
        if self._decision_is_agent(self.game_state['current_player']): self.history.append(('k', (self.game_state['player_cards'][self.game_state['current_player'].name].copy(), self.current_keep)))
        self.phase = "action"
        self._simulate_turn()

    def _simulate_turn(self):
        players, deck, player_cards, player_deaths, player_coins, current_player = self.game_state.values()
        type = self.current_action[2]

        if self.current_block1[1]:
            if self.current_block2[1]:
                blocker_cards = player_cards[self.current_block1[0]]
                if did_block_1_lie(type, blocker_cards):
                    card_idx = [disp[1] for disp in self.current_dispose if disp[0].name == self.current_block1[0]][0]
                    lose_block(self.current_block1[0], player_cards, card_idx, player_deaths)
                    self._take_action()
                else:
                    card_idx = [disp[1] for disp in self.current_dispose if disp[0].name == self.current_block2[0]][0]
                    lose_block(self.current_block2[0], player_cards, card_idx, player_deaths)
            else:
                if self.current_block1[2]:
                    sender_cards = player_cards[self.current_action[0]]
                    if did_action_lie(type, sender_cards):
                        card_idx = [disp[1] for disp in self.current_dispose if disp[0].name == self.current_action[0]][0]
                        lose_block(self.current_action[0], player_cards, card_idx, player_deaths)
                    else:
                        card_idx = [disp[1] for disp in self.current_dispose if disp[0].name == self.current_block1[0]][0]
                        lose_block(self.current_block1[0], player_cards, card_idx, player_deaths)
                        self._take_action()
        else:
            self._take_action()

        self._update_next_player()

    def _take_action(self):
        players, deck, player_cards, player_deaths, player_coins, current_player = self.game_state.values()
        player1_name, player2_name, type = self.current_action

        if type == 'Income':
            income(player1_name, player_coins)
        elif type == 'Foreign Aid':
            foreign_aid(player1_name, player_coins)
        elif type == 'Tax':
            tax(player1_name, player_coins)
        elif type == 'Steal':
            steal(player1_name, player2_name, player_coins)
        elif type == 'Coup':
            card_idx = [disp[1] for disp in self.current_dispose if disp[0].name == self.current_action[1]][0]
            coup(player1_name, player2_name, player_coins, player_cards, card_idx, player_deaths)
        elif type == 'Assassinate' and len(player_cards[player2_name]) > 0:
            card_idx = [disp[1] for disp in self.current_dispose if disp[0].name == self.current_action[1]][0]
            assassinate(player1_name, player2_name, player_coins, player_cards, card_idx, player_deaths)
        elif type == 'Exchange':
            cards = player_cards[current_player.name].copy()
            cards_idxs = self.current_keep
            exchange(player1_name, player_cards, cards, cards_idxs, deck)

    def _update_next_player(self):
        self.game_state['players'] = self.game_state['players'][1:] + [self.game_state['players'][0]]
        self.game_state['players'] = [p for p in self.game_state['players'] if len(self.game_state['player_cards'][p.name]) > 0]
        self.game_state['player_coins'] = {p : (c if len(self.game_state['player_cards'][p]) > 0 else 0) for p, c in zip(self.game_state['player_coins'].keys(), self.game_state['player_coins'].values())}
        self.game_state['current_player'] = self.game_state['players'][0]

    def _determine_disposers(self):
        disposers = []
        players, deck, player_cards, player_deaths, player_coins, current_player = self.game_state.values()
        action_type = self.current_action[2]
        if self.current_block1[1]:
            if self.current_block2[1]:
                blocker_cards = player_cards[self.current_block1[0]]
                if did_block_1_lie(action_type, blocker_cards):
                    disposers.append([p for p in players if p.name == self.current_block1[0]][0])
                    if action_type in ['Coup', 'Assassinate']:
                        disposers.append([p for p in players if p.name == self.current_action[1]][0])
                else:
                    disposers.append([p for p in players if p.name == self.current_block2[0]][0])
            else:
                if self.current_block1[2]:
                    sender_cards = player_cards[self.current_action[0]]
                    if did_action_lie(action_type, sender_cards):
                        disposers.append([p for p in players if p.name == self.current_action[0]][0])
                    else:
                        disposers.append([p for p in players if p.name == self.current_block1[0]][0])
                        if action_type in ['Coup', 'Assassinate']:
                            disposers.append([p for p in players if p.name == self.current_action[1]][0])
        else:
            if action_type in ['Coup', 'Assassinate']:
                disposers.append([p for p in players if p.name == self.current_action[1]][0])
        return disposers

    def _decision_is_agent(self, player):
        player_names = list(self.game_state['player_deaths'].keys())
        return self.agent_idx == player_names.index(player.name)

    def _debug(self):
        pass
        # print("self.phase =", self.phase)
        # print("self.current_action =", self.current_action)
        # print("self.current_block1 =", self.current_block1)
        # print("self.current_block2 =", self.current_block2)
        # if len(self.current_dispose) > 0:
        #     print("self.current_dispose =", [(disp[0].name, disp[1]) for disp in self.current_dispose])
        # print("self.history =", self.history)
        # print('\n')

    
    def _observation(self):
        return np.concatenate((self._encode_gamestate(), self._encode_history()))
    
    def _encode_gamestate(self):
        """
        Return an np array of size 20 + 11 * n that encodes the known information about the game state.

        10  : one-hot encoding of the cards owned by the player
        10  : one-hot encoding of the cards in hand via an exchange action
        10n : one-hot encoding of the cards owned by other players
        n   : number of coins owned by all players (divided by 12, the max number of possible coins)
        """
        encoding = np.zeros((20 + 11 * self.n,))

        players, deck, player_cards, player_deaths, player_coins, current_player = self.game_state.values()
        player_names = list(player_deaths.keys())
        name = player_names[self.agent_idx]
        our_cards = player_cards[name]

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
        for i in range(self.n):
            player_name = player_names[i]
            encoding[20 + i] = player_coins[player_name] / 12
        # fill [20+n : 20+11n] entries with information about player_deaths
        for i in range(self.n):
            player_name = player_names[i]
            if len(player_deaths[player_name]) > 0:
                encoding[20 + self.n + 10 * i + ROLE_TO_I[player_deaths[player_name][0]]] = 1
            if len(player_deaths[player_name]) > 1:
                encoding[20 + self.n + 10 * i + 5 + ROLE_TO_I[player_deaths[player_name][1]]] = 1

        return encoding

    def _encode_history(self):
        """
        Return an np array of size (26 + 6n) * k that encodes the information from the last k turns.

        k turns:
            4 + 4n : action phase, 4 actions on oneself, 3 actions on other players, sender
            3 + n  : block1 phase, 3 (accept, dispute, block) + n (blocker)
            2 + n  : block2 phase, 2 (accept, dispute) + n (blocker)
            26     : keep phase  , 4 * 5 roles + (4 choose 2)

        self.history is stored as the following:
        [('a', action), ('b1', block1), ('a', action), ('b1', block1), ('k', keep), ...]

        action : ( sender, reciever   , action_type     )
        block1 : ( sender, not accept?, refute or block )
        block2 : ( sender, not accept?, refute          )
        keep   : ( cards , card_idxs                    )

        Note: dispose is not a relevant action to store in the memory. 
        Note: keeps are only stored for the agent
        """
        encoding = np.zeros(((6 * self.n + 35) * self.k,))
        encoded_turns, event_encoding = 0, np.zeros((6 * self.n + 35,))
        for event in reversed(self.history):
            if encoded_turns == self.k:
                return encoding
            if event[0] == 'a':
                event_encoding[0:4+4*self.n] = self._encode_action_event(event[1])
                encoding[(6*self.n+35)*encoded_turns:(6*self.n+35)*(encoded_turns+1)] = event_encoding
                encoded_turns += 1
                event_encoding = np.zeros((6 * self.n + 35,))
            elif event[0] == 'b1':
                event_encoding[4+4*self.n:7+5* self.n] = self._encode_block1_event(event[1])
            elif event[0] == 'b2':
                event_encoding[7+5*self.n:9+6*self.n] = self._encode_block2_event(event[1])
            elif event[0] == 'k':
                event_encoding[9+6*self.n:35+6*self.n] = self._encode_keep_event(event[1])
            else:
                exit(1)
        return encoding

    def _encode_action_event(self, action):
        action_encoding = np.zeros((4*self.n+4,))
        sender, reciever, action_type = action
        player_names = list(self.game_state['player_deaths'].keys())

        # encode the sender
        action_encoding[player_names.index(sender)] = 1

        # encode the action_type along with reciever
        if action_type == 'Income':
            action_encoding[self.n] = 1
        elif action_type == 'Foreign Aid':
            action_encoding[self.n + 1] = 1
        elif action_type == 'Tax':
            action_encoding[self.n + 2] = 1
        elif action_type == 'Exchange':
            action_encoding[self.n + 3] = 1
        elif action_type == 'Steal':
            action_encoding[self.n + 4 + player_names.index(reciever)] = 1
        elif action_type == 'Assassinate':
            action_encoding[2 * self.n + 4 + player_names.index(reciever)] = 1 
        elif action_type == 'Coup':
            action_encoding[3 * self.n + 4 + player_names.index(reciever)] = 1
        else:
            exit(1)

        return action_encoding

    def _encode_block1_event(self, block1):
        block1_encoding = np.zeros((3+self.n,))
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
            player_names = list(self.game_state['player_deaths'].keys())
            block1_encoding[3 + player_names.index(sender)] = 1

        return block1_encoding

    def _encode_block2_event(self, block2):
        block2_encoding = np.zeros((2+self.n,))
        sender, not_accept, _ = block2

        # encode accept / refute
        if not not_accept:
            block2_encoding[0] = 1
        else:
            block2_encoding[1] = 1

        # encode the blocker
        if not_accept:
            player_names = list(self.game_state['player_deaths'].keys())
            block2_encoding[2 + player_names.index(sender)] = 1

        return block2_encoding

    def _encode_keep_event(self, keep):
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
    
    def _decode_action(self, a):
        """
        Return an action, block1, block2, dispose, or keep based on the np array, a.
        
        action_size = 4 + 3 * (n - 1) # 4 actions on oneself, 3 actions on other players
        block_1_size = 3 # accept, dispute, block
        block_2_size = 2 # accept, dispute
        dispose_size = 2 # remove card 1 or card 2
        keep_size = 6 # (4 choose 2)
        """
        player_names = list(self.game_state['player_deaths'].keys())
        if self.phase == "action":
            a = a[0:1+3*self.n]
            possible_actions = generate_all_action(self.game_state['current_player'], self.game_state['players'], self.game_state['player_coins'], self.game_state['player_cards'])
            list_of_players = list([p_name for p_name in self.game_state['player_deaths'].keys() if p_name != player_names[self.agent_idx]])
            idx_to_player = {i : list_of_players[i] for i in range(len(list_of_players))}
            idx_to_type = {0: 'Income', 1: 'Foreign Aid', 2: 'Tax', 3: 'Exchange'}
            for i in range(4, 3 + self.n):
                idx_to_type[i] = 'Steal'
            for i in range(3 + self.n, 2 + 2 * self.n):
                idx_to_type[i] = 'Assassinate'
            for i in range(2 + 2 * self.n, 1 + 3 * self.n):
                idx_to_type[i] = 'Coup'
            action = None
            while action not in possible_actions:
                i = np.argmax(a)
                a[i] = -1 * float('inf')
                type = idx_to_type[i]
                if i > 3:
                    reciever = idx_to_player[(i - 4) % (self.n - 1)]
                else:
                    reciever = player_names[self.agent_idx]
                action = (player_names[self.agent_idx], reciever, type)
            self.current_action = action

        elif self.phase == "block1":
            a = a[1+3*self.n:4+3*self.n]
            possible_blocks = generate_all_blocks(player_names[self.agent_idx], self.current_action)
            block1 = None
            while block1 not in possible_blocks:
                i = np.argmax(a)
                a[i] = -1 * float('inf')
                if i == 0: # accept
                    block1 = (player_names[self.agent_idx], False, None)
                if i == 1: # dispute
                    block1 = (player_names[self.agent_idx], True, True)
                if i == 2: # block
                    block1 = (player_names[self.agent_idx], True, False)
            self.current_block1 = block1

        elif self.phase == "block2":
            a = a[4+3*self.n:6+3*self.n]
            possible_blocks = generate_all_blocks(player_names[self.agent_idx], self.current_action)
            block2 = None
            while block2 not in possible_blocks:
                i = np.argmax(a)
                a[i] = -1 * float('inf')
                if i == 0: # accept
                    block2 = (player_names[self.agent_idx], False, None)
                if i == 1: # dispute
                    block2 = (player_names[self.agent_idx], True, True)  
            self.current_block2 = block2

        elif self.phase == "dispose":
            a = a[6+3*self.n:8+3*self.n]
            i = np.argmax(a)
            if len(self.game_state['player_deaths'][player_names[self.agent_idx]]) > 0: i = 0
            self.current_dispose.append((self.players[self.agent_idx], i))

        elif self.phase == "keep":
            a = a[8+3*self.n:14+3*self.n]
            if len(self.game_state['player_cards'][player_names[self.agent_idx]]) < 4:
                i = np.argmax(a[0:3])
                idx_to_keep = {0: [0], 1: [1], 2: [2]}
                self.current_keep = idx_to_keep[i]
            else:
                i = np.argmax(a)
                idx_to_keep = {0: [0, 1],
                               1: [0, 2],
                               2: [0, 3],
                               3: [1, 2],
                               4: [1, 3],
                               5: [2, 3]}
                self.current_keep = idx_to_keep[i]

        else:
            exit(1)

    
    def _reward(self):
        COIN_VALUE, CARD_VALUE, OPP_CARD_VALUE, WIN_VALUE = self.reward_hyperparameters

        reward = 0

        reward += COIN_VALUE * self.game_state['player_coins'][self.players[self.agent_idx].name]
        reward += CARD_VALUE * len(self.game_state['player_cards'][self.players[self.agent_idx].name])
        reward += OPP_CARD_VALUE * sum([len(self.game_state['player_cards'][self.players[i].name]) for i in range(self.n) if i != self.agent_idx])
        if all(self.players[self.agent_idx].name == p for p in [p.name for p in self.game_state['players']]):
            reward += WIN_VALUE
        elif self.players[self.agent_idx].name not in [p.name for p in self.game_state['players']]:
            reward += -1 * WIN_VALUE

        return reward