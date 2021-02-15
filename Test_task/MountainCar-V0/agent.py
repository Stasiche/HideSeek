import numpy as np
from train import transform_state


class Agent:
    def __init__(self, path='agent.npz'):
        self.qlearning_estimate = np.load(path)['arr_0']
        
    def act(self, state):
        state = transform_state(state)
        return np.argmax(self.qlearning_estimate[state])

    def reset(self):
        pass
