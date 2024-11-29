
from envs.detector_env import DetectorEnv

class EvalEnv(DetectorEnv):

    def pick_sample(self):

        self.episode += 1
        if self.episode >= len(self.dataset):
            self.episode = 0
            payload = self.dataset.iloc[-1]['Payloads']
        else:
            payload = self.dataset.iloc[self.episode]['Payloads']
            
        return payload
