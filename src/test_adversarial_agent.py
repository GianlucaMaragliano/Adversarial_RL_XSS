import pandas as pd
from stable_baselines3 import PPO
from utils.utils import init_argument_parser
from gymnasium.utils.env_checker import check_env

from utils.preprocess import process_payloads
from envs.test_env import TestEnv
import numpy as np
import json 

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

def train_adversarial_agent(opt):
    # torch.manual_seed(opt.seed)
    # torch.cuda.manual_seed_all(opt.seed)   
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(opt.seed)

    with open(opt.config_detector, 'r') as f:
        config_detector = json.load(f)

    test_set = pd.read_csv(opt.testset, on_bad_lines='skip')

    env = TestEnv(config_detector, test_set, opt.endpoint,opt.oracle_guided_reward,max_steps=opt.episode_max_steps, num_actions=opt.n_actions)


    check_env(env)

    
    model = PPO.load(opt.checkpoint, device = "cpu")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=len(test_set), deterministic=True, render=False, reward_threshold=None, return_episode_rewards=False)
    print("Mean reward: ", mean_reward)
    print("Std reward: ", std_reward)
    print("Escape Rate: ", env.asr)
    print("Detection Rate: ", env.dr)
    #save to results json file
    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "escape_rate": env.asr,
        "detection_rate": env.dr
    }
    folder = '/'.join(opt.checkpoint.split('/')[:-1])
    with open(f"{folder}/results.json", 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    #save success_test_set to csv
    env.successes_test_set.to_csv(f"{folder}/successes_test_set.csv", index=False)
    #save empirical_study_set to csv
    env.empirical_study_set.to_csv(f"{folder}/empirical_study_set.csv", index=False)



def add_parse_arguments(parser):

    parser.add_argument('--testset', type=str, required = True, help='Training dataset')
    parser.add_argument('--config_detector', type=str, required=True, help='path of the config of the detector')
    parser.add_argument('--checkpoint', type=str, required=True, help='path of the zip file of the agent')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--endpoint', type=str, default='http://127.0.0.1:5555/vuln_backend/1.0/endpoint/', help='Endpoint of the backend used by the oracle')


    #hyperparameters
    parser.add_argument('--episode_max_steps', type=int, default=15, help='max number of steps per episode')
    parser.add_argument('--n_actions', type=int, default=27, help='number of available actions')
    parser.add_argument('--oracle_guided_reward', action = "store_true", help='Use oracle guided reward')


    return parser
    

def main():
    opt = init_argument_parser(add_parse_arguments)
    train_adversarial_agent(opt)

if __name__ == '__main__':
    main()