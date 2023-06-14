import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from player import *
from game import *

I_TO_TYPE = {0: 'Income', 1: 'Foreign Aid', 2: 'Tax', 3: 'Steal', 4: 'Coup', 5: 'Assassinate', 6: 'Exchange'}
TYPE_TO_I = {'Income': 0, 'Foreign Aid': 1, 'Tax': 2, 'Steal': 3, 'Coup': 4, 'Assassinate': 5, 'Exchange': 6}
ROLE_TO_I = {'Duke' : 0, 'Assassin': 1, 'Captain': 2, 'Ambassador': 3, 'Contessa': 4}

def state_to_input(game_state, history, name, role_to_i=ROLE_TO_I):
    """
    Returns a one-hot-encoding of the game_state and history to be used as the input of the model. game_state and history
    are used to get the information required as input.

    input will be a Tensor of shape (10 + 11n)

    The first 10 entries will be in 2 groups of 5, representing the cards the player has
    The next n entries will be the number of coins each player has (normalized by dividing by 12, the max number of possible coins)
    The next 10n entries will be in 2n groups of 5, representing the dead cards each player has revealed
    """
    players, deck, player_cards, player_deaths, player_coins, current_player = game_state['players'], game_state['deck'], game_state['player_cards'], game_state['player_deaths'], game_state['player_coins'], game_state['current_player']
    our_cards = player_cards[name]
    n = len(player_deaths.keys())
    player_names = list(player_deaths.keys())

    # initialize input of zeros
    input = torch.zeros(10 + 11 * n)

    # fill first 10 entries with information about our_cards
    input[role_to_i[our_cards[0]]] = 1
    if len(player_cards[name]) > 1:
        input[role_to_i[our_cards[1]] + 5] = 1

    # fill next n entries with information about player_coins
    for i in range(n):
        player_name = player_names[i]
        input[10 + i] = player_coins[player_name] / 12

    # fill next 10n entries with information about player_deaths
    for i in range(n):
        player_name = player_names[i]
        if len(player_deaths[player_name]) > 0:
            input[10 + n + 10 * i + role_to_i[player_deaths[player_name][0]]] = 1
        if len(player_deaths[player_name]) > 1:
            input[10 + n + 10 * i + 5 + role_to_i[player_deaths[player_name][1]]] = 1

    return input

def output_to_action(output, game_state, name, i_to_type=I_TO_TYPE):
    """
    Returns an action based on the output and game_state. The game_state is used to restrict the output to only pick
    what is allowed within the logic of the game.

    output is a Tensor of shape (n + 7) where n = #players

    The first 7 entries of output represent the choice of action type.
    The next n entries of output represent the choice of reciever.
    """
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    i_to_reciever = {i + 7 : p for i, p in enumerate(game_state['player_deaths'].keys())}

    possible_types = set([action[2] for action in possible_actions])
    for i in range(7):
        if i_to_type[i] not in possible_types:
            output[i] = -1 * float('inf') # might need to make this a PyTorch -infinity
    type = i_to_type[torch.argmax(output[:7]).item()]

    possible_recievers = set([action[1] for action in possible_actions if action[2] == type])
    for i in range(7, len(output)): # might need to change len(output) to be a PyTorch command
        if i_to_reciever[i] not in possible_recievers:
            output[i] = -1 * float('inf') # might need to make this a PyTorch -infinity
    reciever = i_to_reciever[torch.argmax(output[7:]).item() + 7]

    return (name, reciever, type)

def action_to_output(action, game_state):
    """
    Return the simplest one-hot-encoding of the action.

    return output, which is a Tensor of shape (n + 7) where n = #players
    """
    reciever_to_i = {p : i + 7 for i, p in enumerate(game_state['player_deaths'].keys())}
    n = len(game_state['player_deaths'].keys())

    output = torch.zeros(7 + n)

    output[TYPE_TO_I[action[2]]] = 1
    output[reciever_to_i[action[1]]] = 1

    return output

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, name):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.name = name

        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=10000)

    def get_action(self, state, name, epsilon):
        game_state, history = state[0], state[1]

        state = state_to_input(game_state, history, name)

        action_values = self.model.forward(state)

        if torch.rand(1) < epsilon:
            possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
            action = random.choice(possible_actions)
        else:
            action = output_to_action(action_values, game_state, name)

        return action


    def update(self, state, next_state, name, action, reward, done):
        game_state, history = state[0], state[1]
        next_game_state, next_history = next_state[0], next_state[1]

        state_tensor = state_to_input(game_state, history, name)  
        next_state_tensor = state_to_input(next_game_state, next_history, name)  

        # Pass the states through the Q-network to get the predicted Q-values
        predicted_values = self.model.forward(state_tensor)
        predicted_next_values = self.model.forward(next_state_tensor)

        # Get the Q-value of the chosen action
        q_value = predicted_values[action_to_output(action, game_state)]

        # Calculate the target Q-value using the Bellman equation
        if done:
            target_q_value = reward
        else:
            target_q_value = reward + self.gamma * torch.max(predicted_next_values)

        # Calculate the TD error
        td_error = target_q_value - q_value

        # Update the Q-value using gradient descent
        self.optimizer.zero_grad()
        loss = td_error.pow(2).mean()
        loss.backward()
        self.optimizer.step()

    def replay_experience(self, batch_size, name):
        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)

        # Update the Q-network with the sampled experiences
        for experience in batch:
            state, action, reward, next_state, done = experience
            self.update(state, next_state, name, action, reward, done)

    def add_experience(self, state, action, reward, next_state, done):
        # Add the experience to the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def rl_decision(game_state, history, name):
    return rl_action

RL_FUNCS = {'decision_fn': rl_decision, 'block_fn': income_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

class Environment():
    def __init__(self, name):
        self.name = name
        
        players = [Player('Player 1', RL_FUNCS), Player('Player 2', RANDOM_FUNCS), Player('Player 3', RANDOM_FUNCS)]
        self.game = Game(players)


    def step(self, action):
        """
        Return (next_state, reward, done).
        
        next_state = (next_game_state, next_history)
        reward = self.calculate_reward(state, next_state)
        done = True if agent wins / loses
        """
        global rl_action
        rl_action = action

        game_state = self.game.game_state
        history = self.game.history
        state = (game_state, history)

        self.game.simulate_turn()        
        i = len(self.game.game_state['players']) - 1

        while i > 0 and self.game.game_state['current_player'].name != self.name:
            self.game.simulate_turn()   

        next_game_state = self.game.game_state
        next_history = self.game.history
        next_state = (next_game_state, next_history)

        reward = self.calculate_reward(state, next_state)

        done = self.name in [p.name for p in self.game.game_state['players']]  

        return (next_state, reward, done)

    def calculate_reward(self, state, next_state, COIN_VALUE=1, CARD_VALUE=10, CARD_DIVERSITY_VALUE=5, WIN_VALUE=100):
        """
        Calculate the reward from going from state to next_state. 

        + 1 per change in amount of owned coins
        + 10 per change in amount of opponent's cards
        - 10 if you lose a card
        + 5 if 2 of the same card is diversified via exchange
        + 100 if win
        - 100 if lose
        """
        game_state, history = state
        next_game_state, next_history = next_state

        reward = 0

        change_in_coins = next_game_state['player_coins'][self.name] - game_state['player_coins'][self.name]
        reward += COIN_VALUE * change_in_coins

        change_in_opponents_cards = sum([len(next_game_state['player_cards'][p]) for p in next_game_state['player_cards'].keys() if p != self.name]) - sum([len(game_state['player_cards'][p]) for p in game_state['player_cards'].keys() if p != self.name])
        change_in_owned_cards = len(next_game_state['player_cards'][self.name]) - len(game_state['player_cards'][self.name])

        reward += -1 * CARD_VALUE * change_in_opponents_cards
        reward += CARD_VALUE * change_in_owned_cards

        change_in_diversity = len(set(next_game_state['player_cards'][self.name])) - len(set(game_state['player_cards'][self.name]))
        reward += CARD_DIVERSITY_VALUE * change_in_diversity

        if all([p == self.name for p in next_game_state['player_cards'].keys()]):
            reward += WIN_VALUE
        elif self.name not in next_game_state['player_cards'].keys():
            reward += -1 * WIN_VALUE

        return reward

    def reset(self):
        """
        Reset the game to an initial game_state and clear the history. Return the initial_state.

        initial_state = (initial_game_state, initial_history)
        """
        players = [Player('Player 1', RL_FUNCS), Player('Player 2', RANDOM_FUNCS), Player('Player 3', RANDOM_FUNCS)]
        self.game = Game(players)

        initial_game_state = self.game.game_state
        initial_history = self.game.history
        initial_state = (initial_game_state, initial_history)

        return initial_state
    
def calc_reward(name, state, next_state, COIN_VALUE=1, CARD_VALUE=10, CARD_DIVERSITY_VALUE=5, WIN_VALUE=100):
    """This is really garbage code, but I'm pretty tired. """
    temp_env = Environment(name)
    return temp_env.calculate_reward(state, next_state, COIN_VALUE=COIN_VALUE, CARD_VALUE=CARD_VALUE, CARD_DIVERSITY_VALUE=CARD_DIVERSITY_VALUE, WIN_VALUE=WIN_VALUE)