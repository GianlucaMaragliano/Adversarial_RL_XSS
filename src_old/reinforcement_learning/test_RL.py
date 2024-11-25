import pandas as pd
import torch
from stable_baselines3 import PPO

from src.detection_models.utils.general import process_payloads
from src.reinforcement_learning.RL import DetectorEnv

if __name__ == '__main__':

    # test_set = pd.read_csv("../../data/test.csv")
    # test_set = test_set[test_set['Class'] == "Malicious"]
    test_set = pd.read_csv("../../data/adv_xss.txt", header=None, names=['Payloads'], on_bad_lines='skip')
    episodes = test_set.shape[0]

    print("Number of episodes:", episodes)

    env = DetectorEnv(test=True)

    PPO_path = "../../models/PPO_model"
    model = PPO.load(PPO_path+"_test", env=env)

    mutated_attacks_df = pd.DataFrame(columns=['Initial Payload', 'Initial Payload Tokenized', 'Mutated Payload', 'Mutated Payload Tokenized', 'Not Detected'])
    init_payloads = []
    init_payloads_tokenized = []
    mutated_payloads = []
    mutated_payloads_tokenized = []
    detected = []

    breakthrough = 0
    for episode in range(episodes):
        obs = env.reset()
        obs = obs[0]
        xss_df = pd.DataFrame([env.state], columns=['Payloads'])
        xss_df['Class'] = "Malicious"
        done = False
        score = 0

        init_payload = env.state


        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            score += reward
        if done:
            cnn_prediction = torch.round(env.last_cnn_score)
            mlp_prediction = torch.round(env.last_mlp_score)
            if cnn_prediction == 0 or mlp_prediction == 0:
                breakthrough += 1
                detected.append(True)
            else:
                detected.append(False)
            init_payloads.append(init_payload)
            init_payloads_tokenized.append(process_payloads(xss_df, env.common_tokens)[1])
            xss_df = pd.DataFrame([env.state], columns=['Payloads'])
            xss_df['Class'] = "Malicious"
            mutated_payloads.append(env.state)
            mutated_payloads_tokenized.append(process_payloads(xss_df, env.common_tokens)[1])
    print(f"Breakthrough: {breakthrough}/{episodes}, {breakthrough/episodes*100:.2f}%")
    print()

    mutated_attacks_df['Initial Payload'] = init_payloads
    mutated_attacks_df['Initial Payload Tokenized'] = init_payloads_tokenized
    mutated_attacks_df['Mutated Payload'] = mutated_payloads
    mutated_attacks_df['Mutated Payload Tokenized'] = mutated_payloads_tokenized
    mutated_attacks_df['Not Detected'] = detected

    # mutated_attacks_df.to_csv("../../data/mutated_attacks_test.csv", index=False)
