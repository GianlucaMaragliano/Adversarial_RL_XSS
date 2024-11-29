from abc import ABC
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
from envs.detector_env import DetectorEnv
from models.CNN import CNNDetector
from models.MLP import MLPDetector
from models.LSTM import LSTMDetector
from utils.preprocess import process_payloads

from utils import mutators

class TrainEnv(DetectorEnv):

    def pick_sample(self):
        # Randomly select an XSS example from the training set
        payload = self.dataset.sample(1).iloc[0]['Payloads']
            
            
        return payload

