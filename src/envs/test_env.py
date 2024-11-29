
from envs.eval_env import EvalEnv
import pandas as pd

class TestEnv(EvalEnv):

    def __init__(self, config, dataset, max_steps= 15, num_actions = 27):
        super().__init__(config, dataset, max_steps, num_actions)
        self.successes = 0
        self.failures = 0
        self.success_payloads = pd.DataFrame(columns=['Payloads',"Original", "Class"])

    def step(self, action):
        observation, reward, self.done, _, _ = super().step(action)
        if self.done:
            if self.success:
                self.successes += 1
                self.success_payloads = pd.concat([self.success_payloads, pd.DataFrame([[self.payload, self.original_payload, "Malicious"]], columns=['Payloads',"Original", "Class"])])
            else:
                self.failures += 1
        return observation, reward, self.done, False, {}
    
    @property
    def asr(self):
        return self.successes/(self.successes+self.failures)
    
    @property
    def dr(self):
        return self.failures/(self.successes+self.failures)
    
    @property
    def empirical_study_set(self):
        return self.success_payloads[['Payloads', 'Original']]
    
    @property
    def successes_test_set(self):
        return self.success_payloads[['Payloads', 'Class']]
