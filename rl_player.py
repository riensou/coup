import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from player import *

I_TO_TYPE = {0: 'Income', 1: 'Foreign Aid', 2: 'Tax', 3: 'Steal', 4: 'Coup', 5: 'Assassinate', 6: 'Exchange'}
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
    player_names = player_deaths.keys()

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
    i_to_reciever = {i : p for i, p in enumerate(game_state['player_deaths'].keys())}

    possible_types = set([action[2] for action in possible_actions])
    for i in range(7):
        if i_to_type[i] not in possible_types:
            output[i] = -1 * float('inf') # might need to make this a PyTorch -infinity
    type = i_to_type(torch.argmax(output[:7]))

    possible_recievers = set([action[1] for action in possible_actions if action[2] == type])
    for i in range(7, len(output)): # might need to change len(output) to be a PyTorch command
        if i_to_reciever[i] not in possible_recievers:
            output[i] = -1 * float('inf') # might need to make this a PyTorch -infinity
    reciever = i_to_reciever(torch.argmax(output[7:]))

    return (name, reciever, type)

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=10000)

    def get_action(self, state, name, epsilon):
        game_state, history = state[0], state[1]

        state = state_to_input(game_state, history, name)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        action_values = self.model.forward(state_tensor)

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
        q_value = predicted_values[action]

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

class Environment():
    def __init__(self):
        pass

    def step(self, action):
        """
        Return (next_state, reward, done).
        
        next_state = (next_game_state, next_history)
        reward = TBD
        done = True if agent wins / loses
        """
        pass

    def calculate_reward(self, state, next_state):
        """
        Calculate the reward from going from state to next_state. 

        + 1 per change in amount of owned coins
        + 10 per change in amount of opponent's cards
        + 5 if 2 of the same card is diversified via exchange
        + 100 if win
        - 100 if lose
        """
        pass

    def reset(self):
        """
        Reset the game to an initial game_state and clear the history. Return the initial_state.

        initial_state = (initial_game_state, initial_history)
        """
        pass