import gymnasium
import os
import argparse

from coup.coup import Coup

from coup.utils import *
from coup.player import *

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def eval(args):
    algorithm_name = args['model'][args['model'].find('/')+1:args['model'].find('_')]
    if algorithm_name == "PPO": algorithm = PPO
    elif algorithm_name == "A2C": algorithm = A2C
    elif algorithm_name == "DDPG": algorithm = DDPG
    elif algorithm_name == "TD3": algorithm = TD3
    elif algorithm_name == "SAC": algorithm = SAC
    else:
        print("Error: Invalid DRL Algorithm specified")
        return

    model_path = f"{args['model']}"

    load_env = DummyVecEnv([lambda: Monitor(Coup())])
    model = algorithm.load(model_path, env=load_env)

    env = Coup()
    for archetypes in [(RANDOM_FUNCS, "random"), (TRUTH_FUNCS, "truth"), (GREEDY_FUNCS, "greedy"), (INCOME_FUNCS, "income")]:
        total_return = 0
        archetype, name = archetypes
        for _ in range(args['eval_episodes']):
            obs, _ = env.reset(options = {'players': [Player(f"Player {i+1}", archetype) for i in range(4)], 'agent_idx': 1, 'reward_hyperparameters': [0.1, 0.3, -0.25, 1]})
            done = False
            while not done:
                action, _ = model.predict(obs)
                obs, rewards, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_return += rewards
            env.close()
        print(name, "average return:", total_return / args['eval_episodes'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained agent in an environment')

    with open('recent_model.txt', 'r') as file:
        recent_model = file.read().strip()

    parser.add_argument('--model', '-m', type=str, default=recent_model, help='which saved model to evaluate')
    parser.add_argument('--eval_episodes', '-ee', type=int, default=1000, help='the number of episodes to evaluate on')
    args = parser.parse_args()

    eval(vars(args))