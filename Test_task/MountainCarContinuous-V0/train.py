import numpy as np
import os

import torch
from torch import optim
import copy
import random
import gym
from buffer import Buffer
from dqn import DQN

GAMMA = 0.98


def evaluate_policy(agent, episodes=20, seed=None):
    env = gym.make("MountainCarContinuous-v0")
    if seed is not None:
        env.seed(seed)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        while not done:
            state, reward, done, _ = env.step([(agent.act(state)-agent.bnh)/agent.bnh])
            # env.render()
            total_reward += reward
        returns.append(total_reward)
    env.close()
    return returns


def get_next_action(state, eps, model, steps, nbh, device):
    if random.random() < eps / (steps + 1):
        return random.randint(0, nbh*2-1)
    return model(torch.tensor(state).to(device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()


def main():
    if not os.path.exists('agents'):
        os.mkdir('agents')

    env = gym.make("MountainCarContinuous-v0")
    device = torch.device("cuda")
    target_update = 1000
    batch_size = 128

    buckets_number_h = 12
    eps = 0.1
    transitions = 100001

    memory = Buffer(int(1e5))
    m = DQN(device, GAMMA, buckets_number_h)
    optimizer = optim.Adam(m.model.parameters(), lr=1e-3)

    state = env.reset()
    best_reward = -np.inf
    over_ninety_flag = False
    for tr in range(transitions):
        total_reward = 0
        steps = 0

        action = get_next_action(state, eps, m.model, steps, buckets_number_h, device)
        next_state, reward, done, _ = env.step([(action-buckets_number_h)/buckets_number_h])
        if not over_ninety_flag:
            reward += 300 * (GAMMA * abs(next_state[1]) - abs(state[1]))

        memory.push((state, action, reward, next_state, done))

        if not done:
            state = next_state
        else:
            state = env.reset()
            done = False

        if tr > batch_size:
            m.update(memory.sample(batch_size), optimizer)

        if tr % target_update == 0:
            m.target_model = copy.deepcopy(m.model)

            state = env.reset()
            total_reward = 0
            while not done:
                action = get_next_action(state, 0, m.target_model, steps, buckets_number_h, device)
                state, reward, done, _ = env.step([(action-buckets_number_h)/buckets_number_h])
                total_reward += reward

            done = False
            state = env.reset()

        if (tr + 1) % (transitions//100) == 0:
            rewards = evaluate_policy(m, 5)
            cur_rew = np.mean(rewards)
            print(f"Step: {tr+1}, Reward mean: {cur_rew}, Reward std: {np.std(rewards)}")
            if cur_rew > 90:
                over_ninety_flag = True
            if cur_rew > best_reward:
                best_reward = cur_rew
                m.save(os.path.join('agents', 'agent_0'))
