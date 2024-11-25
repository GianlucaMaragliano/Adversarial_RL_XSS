import gymnasium as gym
import pandas as pd
import torch
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict

import numpy as np
import random

from src.detection_models.XSS_dataset import XSSDataset
from src.detection_models.classes.CNN import CNNDetector
from src.detection_models.classes.MLP import MLPDetector
from src.detection_models.utils.general import process_payloads

from src.reinforcement_learning import mutators

root_dir = "../../reproduction"

def load_detection_models():
    vector_size = 8

    train_set = pd.read_csv("../../data/adv_xss.txt", header=None, names=['Payloads'], on_bad_lines='skip')
    _, train_cleaned_tokenized_payloads = process_payloads(train_set)

    sorted_tokens = pd.read_csv(root_dir + "/data/vocabulary.csv")['tokens'].tolist()
    vocab_size = len(sorted_tokens)

    CNN_model = CNNDetector(vocab_size, vector_size)
    CNN_checkpoint = torch.load(root_dir+"/models/CNN_detector.pth")
    CNN_model.load_state_dict(CNN_checkpoint['model_state_dict'])

    MLP_model = MLPDetector(vocab_size, vector_size)
    MLP_checkpoint = torch.load(root_dir+"/models/MLP_detector.pth")
    MLP_model.load_state_dict(MLP_checkpoint['model_state_dict'])

    return CNN_model, MLP_model, sorted_tokens


class DetectorEnv(Env):
    def __init__(self, test=False):
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
        # Load the detection models
        self.CNN_model, self.MLP_model, self.common_tokens = load_detection_models()
        # Current step
        self.current_step = 0
        # Initial last scores for each model is 1 = Malicious
        self.last_cnn_score, self.last_mlp_score = 1, 1

        # Load the training set and test set
        # df = pd.read_csv("../../data/train.csv")
        df = pd.read_csv("../../data/adv_xss.txt", header=None, names=['Payloads'], on_bad_lines='skip')
        # self.train_set = df[df['Class'] == "Malicious"].sample(frac=1, random_state=42)
        # self.train_set = df.sample(frac=1, random_state=42)
        self.train_set = df

        # df = pd.read_csv("../../data/test.csv")
        # self.test_set = df[df['Class'] == "Malicious"]
        self.test_set = df
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

        cnn_output = self.CNN_model(xss_data)
        mlp_output = self.MLP_model(xss_data)
        cnn_prediction = torch.round(cnn_output)
        mlp_prediction = torch.round(mlp_output)

        embedded = self.CNN_model.embedding(xss_data).detach().numpy()

        self.last_cnn_score, self.last_mlp_score = cnn_output, mlp_output

        # print(self.max_steps)
        # print(self.state)
        # print(process_xss_example)
        # print(self.actions_taken)
        # print(f"CNN Prediction: {cnn_output}, MLP Prediction: {mlp_output}")
        # print()

        outcome = not (cnn_prediction == 1 and mlp_prediction == 1)

        if not outcome:
            reward = 0
            done = False
        else:
            # print('Success with payload:', mutated_xss_example)
            reward = 1
            done = True

        self.max_steps -= 1
        if self.max_steps == 0:
            done = True

        scores = np.array([mlp_output.item(), cnn_output.item()], dtype=np.float32)
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
                self.state = self.train_set.sample(1).iloc[0]['Payloads']
            else:
                # Test on every XSS example in the test set
                self.episode += 1
                self.state = self.test_set.iloc[self.episode]['Payloads']

        self.max_steps = 15
        self.last_cnn_score, self.last_mlp_score = 1, 1
        self.current_step = 0
        self.actions_taken = np.zeros(self.max_steps, dtype=np.int32)
        xss_df = pd.DataFrame([self.state], columns=['Payloads'])
        xss_df['Class'] = "Malicious"
        _, process_xss_example = process_payloads(xss_df, self.common_tokens)
        xss_dataset = XSSDataset(process_xss_example, xss_df['Class'])
        xss_data = xss_dataset[0][0][None, ...]
        string = self.CNN_model.embedding(xss_data).detach().numpy()
        # obs = {"confidence": np.array([1.0, 1.0], dtype=np.float32), "actions": self.actions_taken, "string": string}
        obs = string
        return obs, {}


