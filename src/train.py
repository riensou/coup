import gymnasium
import os
import random
import string
import argparse

from coup.coup import Coup
from coup.logger import TensorboardCallback

from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def train(args):
    environment = "Coup"

    if args['algorithm'] == "PPO": algorithm = PPO
    elif args['algorithm'] == "A2C": algorithm = A2C
    elif args['algorithm'] == "DDPG": algorithm = DDPG
    elif args['algorithm'] == "TD3": algorithm = TD3
    elif args['algorithm'] == "SAC": algorithm = SAC
    else:
        print("Error: Invalid DRL Algorithm specified")
        return

    id = ''.join(random.choices(string.ascii_letters, k=6))

    full_id = args['algorithm'] + '_' + environment + '_' + id

    models_dir = f"models/{full_id}"
    logdir = "data"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    train_env = Monitor(Coup(), info_keywords=("is_success",))
    train_env = DummyVecEnv([lambda: train_env])

    model = algorithm("MlpPolicy", train_env, verbose=1, tensorboard_log=logdir if args["log"] else None)

    for i in range(1, args['num_iters']):
        model.learn(total_timesteps=args['steps_per_iter'], reset_num_timesteps=False, tb_log_name=full_id, callback=TensorboardCallback())
        if args["log"]:
            model.save(f"{models_dir}/{args['steps_per_iter']*i}")
            with open('recent_model.txt', 'w') as file:
                file.write(f"{models_dir}/{args['steps_per_iter']*i}")
    train_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an agent in an environment using a DRL algorithm from stable baselines 3')

    parser.add_argument('--log', type=bool, default=True, help='record logs to tensorboard')
    parser.add_argument('--algorithm', '-a', type=str, default="PPO", help='which DRL algorithm to use for training')
    parser.add_argument('--steps_per_iter', '-spi', type=int, default=10000, help='the number of timesteps for each saved model')
    parser.add_argument('--num_iters', '-ni', type=int, default=100, help='the number of iterations')
    args = parser.parse_args()

    train(vars(args))