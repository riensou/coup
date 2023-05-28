ROLES = ['Duke', 'Assassin', 'Captain', 'Ambassador', 'Contessa']

LIE_BLOCKABLE = {'Income': False, 'Foreign Aid': False, 'Tax': True, 'Steal': True, 'Coup': False, 'Assassinate': True, 'Exchange': True, 'Block': True}
ROLE_BLOCKABLE = {'Income': True, 'Foreign Aid': True, 'Tax': False, 'Steal': True, 'Coup': False, 'Assassinate': True, 'Exchange': False, 'Block': False}

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

# -- Generate Moves -- #
def generate_all_action(current_player, players, player_coins, player_cards):
    """
    Return all possible actions of the form (p1, p2, type) where

    p1 = current_player
    p2 = any other player that is still alive
    type = the type of action 
    """
    p1 = current_player.name
    other_players = [p2.name for p2 in players if p2.name != p1 and player_cards[p2.name] > 0]

    possible_actions = []

    if player_coins[p1] >= 10:
        return [(p1, p2, 'Coup') for p2 in other_players]
    if player_coins[p1] >= 3:
        possible_actions += [(p1, p2, 'Assassinate') for p2 in other_players]
    if player_coins[p1] >= 7:
        possible_actions += [(p1, p2, 'Coup') for p2 in other_players]
    
    possible_actions += [(p1, p1, 'Income'), (p1, p1, 'Foreign Aid'), (p1, p1, 'Tax'), (p1, p1, 'Exchange')]

    possible_actions += [(p1, p2.name, 'Steal') for p2 in players if p2.name != p1 and player_coins[p2.name] > 0]

    return possible_actions

def generate_all_blocks(current_player, action):
    p1 = current_player.name

    possible_blocks = [(p1, False, None)]

    if ROLE_BLOCKABLE[action[2]]:
        possible_blocks += [(p1, True, False)]

    if LIE_BLOCKABLE[action[2]]:
        possible_blocks += [(p1, True, True)]

    return possible_blocks
    
    
