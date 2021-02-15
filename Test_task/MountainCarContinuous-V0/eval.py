from train import evaluate_policy
from agent import Agent
import numpy as np
import os
import torch


def eval_main(agents_lst):
    # agents_lst = ['agent_1']
    bnh = 12
    for _ in range(5):
        seed = np.random.randint(1, int(1e5))
        for path in map(lambda x: os.path.join('agents', x), agents_lst):
            device = torch.device("cuda")
            ag = Agent(device, bnh, path)
            rew = evaluate_policy(ag, seed=seed)
            # print(os.path.split(path)[1], np.mean(rew))
            print(os.path.split(path)[1], np.mean(rew), np.std(rew), rew)
        print('_____________________')