import random
from itertools import cycle

class Game:

    def __init__(self):
        # Define the roles and their abilities
        self.roles = {
            'Duke': 'Tax',
            'Assassin': 'Assassinate',
            'Captain': 'Steal',
            'Ambassador': 'Exchange',
            'Contessa': None
        }

        # Define the players (change as needed)
        self.players = ['Player 1', 'Player 2', 'Player 3']
        # Define the initial influence (number of character cards) for each player
        self.player_influence = {player: 2 for player in self.players}
        # Define the player's coins (change as needed)
        self.player_coins = {player: 2 for player in self.players}
        # Define the deck of character cards
        self.deck = list(self.roles.keys()) * 3
        # Shuffle the deck
        random.shuffle(self.deck)
        # Deal two character cards to each player
        self.player_cards = {}
        if len(self.deck) >= len(self.players) * 2:
            for player in self.players:
                self.player_cards[player] = [self.deck.pop(), self.deck.pop()]
        else:
            print("Not enough cards in the deck to deal to each player.")
            exit()
        # Define turn order
        self.players_turn = cycle(self.players)

    # Function to get whose turn it is
    def next_turn(self):
        return next(self.players_turn)

    # Function to display the current state of the game
    def display_game_state(self):
        print('--- Game State ---')
        for player in self.players:
            print(f'{player}: Influence={self.player_influence[player]}, Coins={self.player_coins[player]}, Cards={self.player_cards[player]}')
        print('-----------------')

    # Simulate a player's turn
    def simulate_player_turn(self, player):
        # Choose an action randomly for demonstration purposes
        possible_actions = ['Income', 'Foreign Aid', 'Tax', 'Exchange', 'Steal']
        if self.player_coins[player] >= 3:
            possible_actions.append('Assassinate')
        if self.player_coins[player] >= 7:
            possible_actions.append('Coup')
        if self.player_coins[player] >= 10:
            possible_actions = ['Coup']
        if self.player_influence[player] == 0:
            possible_actions = []

        if possible_actions:
            action = random.choice(possible_actions)
            print(f"Player {player}'s Turn:")
            print(f"Selected Action: {action}")
        else:
            return

        # Perform the chosen action
        if action == 'Income':
            self.player_coins[player] += 1
        elif action == 'Foreign Aid':
            self.player_coins[player] += 2
            
        elif action == 'Tax':
            self.player_coins[player] += 3

        elif action == 'Assassinate':
            # Choose a random target player to assassinate
            remaining_players = [p for p in self.players if p != player or self.player_influence[p] == 0]  # Exclude the current player and those who have no influence
            target_player = random.choice(remaining_players)
            print(f"Assassinating Player: {target_player}")

            # Pay the assassination cost and force target player to lose influence
            self.player_coins[player] -= 3
            self.player_influence[target_player] -= 1

            # Randomly select a card to lose
            lost_card = random.choice(self.player_cards[target_player])
            self.player_cards[target_player].remove(lost_card)
            if self.player_influence[target_player] == 0:
                # If the target player now has zero influence, they lose the game
                print(f"Player {target_player} has lost the game.")
                self.player_coins[target_player] = 0

        elif action == 'Exchange':
            # Perform the exchange action by drawing two new cards and choosing which cards to keep
            if len(self.deck) >= 2:
                new_cards = [self.deck.pop(), self.deck.pop()]
                self.player_cards[player] += new_cards
                random.shuffle(self.player_cards[player])
                self.deck += self.player_cards[player][:2]
                self.player_cards[player]= self.player_cards[player][2:]
                print(f"Player {player} exchanged cards. New cards: {self.player_cards[player]}")
            else:
                print("Not enough cards in the deck to perform the exchange.")

        elif action == 'Steal':
            # Choose a random target player to steal from
            remaining_players = [p for p in self.players if p != player or self.player_coins[p] == 0]  # Exclude the current player and those who have no money
            target_player = random.choice(remaining_players)
            print(f"Stealing from Player: {target_player}")

            # Perform the steal action, taking 2 coins from the target player if they have at least 2 coins
            self.player_coins[player] += min(2, self.player_coins[target_player])
            self.player_coins[target_player] -= min(2, self.player_coins[target_player])

        elif action == 'Coup':
            # Choose a random target player to coup
            remaining_players = [p for p in self.players if p != player or self.player_influence[p] == 0]  # Exclude the current player and those who have no influence
            target_player = random.choice(remaining_players)
            print(f"Couping Player: {target_player}")

            # Pay the coup cost and force target player to lose influence
            self.player_coins[player] -= 7
            self.player_influence[target_player] -= 1

            # Randomly select a card to lose
            lost_card = random.choice(self.player_cards[target_player])
            self.player_cards[target_player].remove(lost_card)
            if self.player_influence[target_player] == 0:
                # If the target player now has zero influence, they lose the game
                print(f"Player {target_player} has lost the game.")
                self.player_coins[target_player] = 0


        # Display the updated game state
        self.display_game_state()

    def play(self):
        while(len([influence for influence in self.player_influence.values() if influence > 0]) > 1):
            self.simulate_player_turn(self.next_turn())


# Example usage:
game = Game()
game.display_game_state()
game.play()