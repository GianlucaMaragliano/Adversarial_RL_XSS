from abc import ABC, abstractmethod
from calendar import c
import gymnasium as gym
import pandas as pd
import torch
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, MultiDiscrete
import copy

import numpy as np
import random

from datasets.xss_dataset import XSSDataset
from models.CNN import CNNDetector
from models.MLP import MLPDetector
from models.LSTM import LSTMDetector
from utils.preprocess import process_payloads

from utils import mutators

class DetectorEnv(Env, ABC):

    @abstractmethod
    def pick_sample(self):
        pass

    def __init__(self, config, dataset, max_steps= 15, num_actions = 27):

        self.num_actions = num_actions
        # There are 27 possible actions
        self.action_space = Discrete(self.num_actions )
        # Maximum number of steps
        self.max_steps = max_steps
        # Array of actions taken
        self.actions_taken = np.zeros(self.max_steps, dtype=np.int32)
        self.episode = -1

        # Observation space represents confidence scores of the CNN and MLP models and past action list
        # self.observation_space = Dict({"confidence": Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        #                                "actions": Box(low=0, high=27, shape=(self.max_steps,), dtype=np.int32),
        #                                "string": Box(low=-10, high=10, shape=(1, 30, 8), dtype=np.float32)})
        self.observation_space = Box(low=0, high=self.num_actions , shape=(self.max_steps,), dtype=np.int32)

        # Current state is the current XSS example status
        self.state = np.zeros(self.max_steps, dtype=np.int32)
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
        self.common_tokens = sorted_tokens
        self.model = model_architecture(vocab_size, config["embedding_dim"])
        self.model.load_state_dict(torch.load(config["checkpoint"]))
        # Current step
        self.current_step = 0
        # Initial last scores for each model is 1 = Malicious
        self.last_score =  1

        # Load the dataset
    
        self.dataset = dataset
        #self.payload = self.pick_sample()
        #copy the string in self.payload in self.original_payload
        #self.original_payload = copy.copy(self.payload)
        self.remaining_steps = max_steps
        self.done = False
        self.success = False

    def calculate_reward(self, xss_df):
        xss_df['Class'] = "Malicious"
        _, process_xss_example = process_payloads(xss_df, self.common_tokens)
        xss_dataset = XSSDataset(process_xss_example, xss_df['Class'])
        xss_data = xss_dataset[0][0][None, ...]
        
        output = self.model(xss_data)
        prediction = torch.round(output)


        self.last_score = output

        outcome = not (prediction == 1)

        if not outcome:
            reward = -1
            self.done = False
            self.success = False
        else:
            #print('Success with payload:', mutated_xss_example)
            reward = 10
            self.done = True
            self.success = True


        self.remaining_steps -= 1
        if self.remaining_steps == 0:
            self.done = True
            self.success = False
        return reward


    def step(self, action):
        action_name = "action_" + str(action + 1)
        mutator = getattr(mutators, action_name)
        try:
            #print(f"Original XSS example: {self.payload}")

            mutated_xss_example = mutator(self.payload)
            #print(f"Mutated XSS example: {mutated_xss_example}")
            self.payload = mutated_xss_example
            self.actions_taken[self.current_step] = action+1
            #print(action)
            self.state[self.current_step] = action
            #print(self.state)
            self.current_step += 1
        except Exception as e:
            print(f"Error: {e}")
            mutated_xss_example = self.payload

        xss_df = pd.DataFrame([mutated_xss_example], columns=['Payloads'])
        
        reward = self.calculate_reward(xss_df)
       
        # observation = {"confidence": scores, "actions": self.actions_taken, "string": embedded}
        # observation = {"actions": self.actions_taken}
        # print(self.episode, self.original_payload)
        observation = self.state
        return observation, reward, self.done, False, {}

    def reset(self, init_state=None, seed=None, options=None, test=False):
        super().reset(seed=seed)
        self.payload = self.pick_sample()
        self.original_payload = copy.copy(self.payload)
        self.state = np.zeros(self.max_steps, dtype=np.int32)
        self.remaining_steps = self.max_steps
        self.last_score = 1
        self.current_step = 0
        self.actions_taken = np.zeros(self.max_steps, dtype=np.int32)
        self.done = False
        self.success = False
        # obs = {"confidence": np.array([1.0, 1.0], dtype=np.float32), "actions": self.actions_taken, "string": string}
        return self.state, {}