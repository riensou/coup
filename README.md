![coup](https://github.com/riensou/coup/assets/coup.jpeg)


Python implentation of the popular bluffing card game, _Coup_.

For a description of the rules, click [here](https://www.ultraboardgames.com/coup/game-rules.php).


**Player Descriptions:**
1. Random
- all logic functions make completely random choices

2. Truth Teller
- only chooses moves that never bluff based on its own cards

3. User
- chooses moves based on user input to the terminal

4. Income
- always takes income or assinates/coups whenever possible

5. Neural Network
- WIP


**Running the program:**
1. Create player objects using the various FUNCS from player.py
2. Put all players into the players list and create the game object.
3. Run python3 game.py


**Missing Features / Bugs:**
* Blocking the 'Steal' action is ambiguous, and doesn't force the player to specifically claim 'Captain' or 'Ambassador'
* Winning a challenge does not cause the player to reshuffle their card back into the deck
* There is no implementation for a GUI yet
