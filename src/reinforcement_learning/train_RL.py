import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from src.detection_models.utils.general import process_payloads
from src.reinforcement_learning.RL import DetectorEnv
from gymnasium.utils.env_checker import check_env


def make_env():
    return DetectorEnv()


if __name__ == '__main__':
    env = make_env()

    episodes = 10
    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0

        while not done:
            action = env.action_space.sample()
            n_state, reward, done, _, info = env.step(action)
            score += reward

        print(f"Episode: {episode + 1}/{episodes}, Score: {score}")

    PPO_path = "../../models/PPO_model"
    SAC_path = "../../models/SAC_model"

    check_env(env)

    # model = SAC("MultiInputPolicy", env, verbose=1)
    model = SAC.load(SAC_path)

    # del model
    # model.learn(total_timesteps=20000)

    # Evaluate the trained model
    mean_reward, _ = evaluate_policy(model, Monitor(env), n_eval_episodes=20)
    print(f"Mean reward: {mean_reward}\n")

    # Save the model

    # model.save(SAC_path)
    # del model
    # model = SAC.load(SAC_path)

    for episode in range(episodes):
        obs = env.reset()
        obs = obs[0]
        print("Initial State: ", env.state)
        xss_df = pd.DataFrame([env.state], columns=['Payloads'])
        xss_df['Class'] = "Malicious"
        print("Tokenized State: ", process_payloads(xss_df, env.common_tokens)[1])
        done = False
        score = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _, info = env.step(action)
            score += reward
        if done:
            xss_df = pd.DataFrame([env.state], columns=['Payloads'])
            xss_df['Class'] = "Malicious"
            print("Tokenized State: ", process_payloads(xss_df, env.common_tokens)[1])
            print("Final State: ", env.state)
            print("Final CNN prediction: ", torch.round(env.last_cnn_score))
            print("Final MLP prediction: ", torch.round(env.last_mlp_score))

        print(f"Episode: {episode + 1}/{episodes}, Score: {score}")
        print()

    mean_reward, _ = evaluate_policy(model, Monitor(env), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}")
