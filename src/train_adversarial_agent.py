from random import seed
import pandas as pd
from stable_baselines3 import PPO
from utils.utils import init_argument_parser
from utils.path_utils import get_last_run_number

from envs.train_env import TrainEnv
from envs.eval_env import EvalEnv
from gymnasium.utils.env_checker import check_env
import json 
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import os
def train_adversarial_agent(opt):
    # torch.manual_seed(opt.seed)
    # torch.cuda.manual_seed_all(opt.seed)   
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(opt.seed)

    with open(opt.config_detector, 'r') as f:
        config_detector = json.load(f)
    
    train_set = pd.read_csv(opt.trainset, on_bad_lines='skip')

    validation_set = pd.read_csv(opt.valset, on_bad_lines='skip')

    train_env = TrainEnv(config_detector,train_set, max_steps=opt.episode_max_steps, num_actions=opt.n_actions)
    check_env(train_env)
    eval_env = EvalEnv(config_detector, validation_set, max_steps=opt.episode_max_steps, num_actions=opt.n_actions)
    check_env(eval_env)

    detector_folder = '/'.join(opt.config_detector.split('/')[:-1])
    save_folder = f"{detector_folder}/{opt.runs_folder}"
    os.makedirs(save_folder, exist_ok=True)
    last_run = get_last_run_number(save_folder)
    save_folder = os.path.join(save_folder, f"run_{last_run + 1}")
    os.makedirs(save_folder, exist_ok=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=save_folder,
                             log_path=save_folder, eval_freq=opt.timesteps/10,
                             n_eval_episodes=len(validation_set),
                             deterministic=True, render=False)
    model = PPO("MlpPolicy", train_env, verbose=1, device="cpu")

    model.learn(total_timesteps=opt.timesteps, callback=eval_callback)

def add_parse_arguments(parser):

    parser.add_argument('--trainset', type=str, required = True, help='Training dataset')
    parser.add_argument('--valset', type=str, required = True, help='Validation dataset')
    parser.add_argument('--config_detector', type=str, required=True, help='path of the config of the detector')
    parser.add_argument('--runs_folder', type=str, default="adversarial_agent", help='Runs Folder')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')


    #hyperparameters
    parser.add_argument('--timesteps', type=int, default=250000, help='number of epochs to train')
    parser.add_argument('--episode_max_steps', type=int, default=15, help='max number of steps per episode')
    parser.add_argument('--n_actions', type=int, default=27, help='number of available actions')


    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    train_adversarial_agent(opt)

if __name__ == '__main__':
    main()