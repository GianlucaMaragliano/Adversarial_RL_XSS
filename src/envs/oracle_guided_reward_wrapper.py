from envs.detector_env import DetectorEnv
from utils.dataset_utils import check_column_validity_with_oracle
from datasets.xss_dataset import XSSDataset

from utils.preprocess import process_payloads
import pandas as pd
import torch

class OracleGuidedRewardWrapper(DetectorEnv):
    def __init__(self, env, endpoint, config, dataset, max_steps= 15, num_actions = 27):
        self.env = env
        self.endpoint = endpoint
        env.__init__(config, dataset, max_steps, num_actions)
    
    def pick_sample(self):
        return self.env.pick_sample()
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self, init_state=None, seed=None, options=None, test=False):
        return self.env.reset(init_state, seed, options, test)
    
    def calculate_reward(self, xss_df):
        xss_df['Class'] = "Malicious"
        xss_df["tokenized"]= process_payloads(xss_df, self.common_tokens)[1]
        valid = (check_column_validity_with_oracle(xss_df, 'reconstructed_str_payload_original', self.endpoint)).map(lambda x: not x)
        #drop the column tokenized
        xss_df = xss_df.drop(columns=['tokenized'])
        if valid:
            return self.env.calculate_reward(xss_df)
        else:
            return -5
