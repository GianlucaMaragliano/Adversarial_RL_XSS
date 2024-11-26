import gymnasium as gym
import pandas as pd
import torch
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict

import numpy as np
import random

from datasets.xss_dataset import XSSDataset
from models.CNN import CNNDetector
from models.MLP import MLPDetector
from models.LSTM import LSTMDetector
from utils.preprocess import process_payloads

from utils import mutators
import json

class DetectorEnv(Env):
    def __init__(self, config, dataset, test=False):
        # There are 27 possible actions
        self.action_space = Discrete(27)
        # Maximum number of steps
        self.max_steps = 15
        # Array of actions taken
        self.actions_taken = np.zeros(self.max_steps, dtype=np.int32)

        # Observation space represents confidence scores of the CNN and MLP models and past action list
        # self.observation_space = Dict({"confidence": Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        #                                "actions": Box(low=0, high=27, shape=(self.max_steps,), dtype=np.int32),
        #                                "string": Box(low=-10, high=10, shape=(1, 30, 8), dtype=np.float32)})
        # self.observation_space = Dict({"actions": Box(low=0, high=27, shape=(self.max_steps,), dtype=np.int32)})

        self.observation_space = Box(low=-10, high=10, shape=(1, 30, 8), dtype=np.float32)
        # Current state is the current XSS example status
        self.state = "https://<script>alert('1')</script>"
        # Load the json file
  
        sorted_tokens = pd.read_csv(config["vocabulary"])['tokens'].tolist()
        if config["model"] == 'mlp':
            model_architecture = MLPDetector
        elif config["model"] == 'cnn':
            model_architecture = CNNDetector
        elif config["model"] == 'lstm':
            model_architecture = LSTMDetector
        else:
            raise ValueError("Model not supported")
        vocab_size = len(sorted_tokens)

        self.model = model_architecture(vocab_size, config["embedding_dim"])
        # Current step
        self.current_step = 0
        # Initial last scores for each model is 1 = Malicious
        self.last_score =  1

        # Load the dataset
    
        self.dataset = pd.read_csv(dataset, header=None, names=['Payloads'], on_bad_lines='skip')

        self.test = test
        self.episode = -1

    def step(self, action):
        action_name = "action_" + str(action + 1)
        mutator = getattr(mutators, action_name)
        try:
            mutated_xss_example = mutator(self.state)
        except Exception as e:
            print(f"Error: {e}")
            mutated_xss_example = self.state

        xss_df = pd.DataFrame([mutated_xss_example], columns=['Payloads'])
        xss_df['Class'] = "Malicious"
        _, process_xss_example = process_payloads(xss_df, self.common_tokens)

        xss_dataset = XSSDataset(process_xss_example, xss_df['Class'])
        xss_data = xss_dataset[0][0][None, ...]

        self.state = mutated_xss_example
        self.actions_taken[self.current_step] = action+1
        self.current_step += 1

        output = self.model(xss_data)
        prediction = torch.round(output)

        embedded = self.model.embedding(xss_data).detach().numpy()

        self.last_score = output

        outcome = not (prediction == 1)

        if not outcome:
            reward = -1
            done = False
        else:
            # print('Success with payload:', mutated_xss_example)
            reward = 10
            done = True

        self.max_steps -= 1
        if self.max_steps == 0:
            done = True

        scores = np.array(output.item(), dtype=np.float32)
        # observation = {"confidence": scores, "actions": self.actions_taken, "string": embedded}
        # observation = {"actions": self.actions_taken}
        observation = embedded
        return observation, reward, done, False, {}

    def reset(self, init_state=None, seed=None, options=None, test=False):
        super().reset(seed=seed)
        self.test = test
        if init_state is not None:
            self.state = init_state
        else:
            if not self.test:
                # Randomly select an XSS example from the training set
                self.state = self.dataset.sample(1).iloc[0]['Payloads']
            else:
                # Test on every XSS example in the test set
                self.episode += 1
                self.state = self.dataset.iloc[self.episode]['Payloads']

        self.max_steps = 15
        self.last_score = 1
        self.current_step = 0
        self.actions_taken = np.zeros(self.max_steps, dtype=np.int32)
        xss_df = pd.DataFrame([self.state], columns=['Payloads'])
        xss_df['Class'] = "Malicious"
        _, process_xss_example = process_payloads(xss_df, self.common_tokens)
        xss_dataset = XSSDataset(process_xss_example, xss_df['Class'])
        xss_data = xss_dataset[0][0][None, ...]
        string = self.model.embedding(xss_data).detach().numpy()
        # obs = {"confidence": np.array([1.0, 1.0], dtype=np.float32), "actions": self.actions_taken, "string": string}
        obs = string
        return obs, {}