from train import evaluate_policy
from agent import Agent
import numpy as np
import os


def eval_main(agents_lst):
    for _ in range(10):
        seed = np.random.randint(1, int(1e5))
        for path in map(lambda x: os.path.join('agents', x)+'.npz', agents_lst):
            ag = Agent(path)
            rew = evaluate_policy(ag, 20, seed=seed)
            print(os.path.split(path[:-4])[1], np.mean(rew), np.round(np.std(rew), 1), rew)
        print('_____________________')