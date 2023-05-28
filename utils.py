ROLES = ['Duke', 'Assassin', 'Captain', 'Ambassador', 'Contessa']

# -- Coin Gain -- #
def income(player_name, player_coins):
    player_coins[player_name] += 1

def foreign_aid(player_name, player_coins):
    player_coins[player_name] += 2

def tax(player_name, player_coins):
    player_coins[player_name] += 3

def steal(player1_name, player2_name, player_coins):
    player_coins[player1_name] += 2
    player_coins[player2_name] -= 2

# -- Card Loss -- #
def coup(player1_name, player2_name, player_coins, player_cards, card_idx, player_deaths):
    player_coins[player1_name] -= 7
    lost_card = player_cards[player2_name].pop(card_idx[0])
    player_deaths[player1_name].append(lost_card)

def assassinate(player1_name, player2_name, player_coins, player_cards, card_idx, player_deaths):
    player_coins[player1_name] -= 3
    lost_card = player_cards[player2_name].pop(card_idx[0])
    player_deaths[player1_name].append(lost_card)

def lose_block(player_name, player_cards, card_idx, player_deaths):
    lost_card = player_cards[player_name].pop(card_idx[0])
    player_deaths[player_name].append(lost_card)

# -- Exchange -- #
def exchange(player_name, player_cards, cards, cards_idxs, deck):
    player_cards[player_name] = cards[cards_idxs]
    for idx in cards_idxs:
        cards.pop(idx)
    deck += cards
