import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from utils.utils import init_argument_parser

from utils.preprocess import process_payloads
from envs.detector_env import DetectorEnv
from gymnasium.utils.env_checker import check_env
import json 
from stable_baselines3.common.callbacks import EvalCallback

def train_adversarial_agent(opt):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(opt.config_detector, 'r') as f:
        config_detector = json.load(f)

    train_env = DetectorEnv(config_detector, opt.trainset)
    check_env(train_env)
    eval_env = DetectorEnv(config_detector, opt.valset, test=True)
    check_env(eval_env)

    detector_folder = '/'.join(opt.config_detector.split('/')[:-1])
    save_folder = f"{detector_folder}/{opt.runs_folder}"
    validation_set = pd.read_csv(opt.valset).sample(frac=1)

    eval_callback = EvalCallback(eval_env, best_model_save_path=save_folder,
                             log_path=save_folder, eval_freq=opt.timesteps/10,
                             eval_episodes=len(validation_set),
                             deterministic=True, render=False)
    model = PPO("MlpPolicy", train_env, verbose=1)

    model.learn(total_timesteps=20000, callback=eval_callback)

def add_parse_arguments(parser):

    parser.add_argument('--trainset', type=str, default="data/RL/train.csv", help='Training dataset')
    parser.add_argument('--valset', type=str, default="data/RL/val.csv", help='Validation dataset')
    parser.add_argument('--config_detector', type=str, required=True, help='path of the config of the detector')
    parser.add_argument('--runs_folder', type=str, default="adversarial_agent", help='Runs Folder')
    parser.add_argument('--vocab_size', type=float, default=0.1, help='Percentage of the most common tokens to keep in the vocab')


    #hyperparameters
    parser.add_argument('--timesteps', type=int, default=200000, help='number of epochs to train')
    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    train_adversarial_agent(opt)

if __name__ == '__main__':
    main()