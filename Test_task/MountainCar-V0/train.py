from gym import make
import numpy as np
import random
import os

GAMMA = 0.98
GRID_SIZE_X = 30
GRID_SIZE_Y = 30


def transform_state(state):
    state = (np.array(state) + np.array((1.2, 0.07))) / np.array((1.8, 0.14))
    x = min(int(state[0] * GRID_SIZE_X), GRID_SIZE_X-1)
    y = min(int(state[1] * GRID_SIZE_Y), GRID_SIZE_Y-1)
    return x + GRID_SIZE_X*y


class QLearning:
    def __init__(self, state_dim, action_dim, alpha=0.1, gamma=0.98):
        self.qlearning_estimate = np.zeros((state_dim, action_dim)) + 2.
        self.alpha = alpha
        self.gamma = gamma

    def update(self, transition):
        state, action, next_state, reward, done = transition
        next_action = self.act(next_state)
        state = transform_state(state)
        next_state = transform_state(next_state)
        self.qlearning_estimate[state, action] = (1-self.alpha)*self.qlearning_estimate[state, action] + \
            self.alpha * (reward + self.gamma*self.qlearning_estimate[next_state, next_action])

    def act(self, state):
        state = transform_state(state)
        return np.argmax(self.qlearning_estimate[state])

    def save(self, path='agent.npz'):
        np.savez(path, self.qlearning_estimate)


def evaluate_policy(agent, episodes=5, seed=None):
    env = make("MountainCar-v0")
    if seed is not None:
        env.seed(seed)
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.
        
        while not done:
            state, reward, done, _ = env.step(agent.act(state))
            # env.render()
            total_reward += reward
        returns.append(total_reward)
    env.close()
    return returns


def main(agent_name, eps_reducer, save_best, reward_shaping_constant):
    if not os.path.exists('agents'):
        os.mkdir('agents')

    env = make("MountainCar-v0")
    env.seed(3)
    ql = QLearning(state_dim=GRID_SIZE_X*GRID_SIZE_Y, action_dim=3)
    eps = 0.1
    transitions = 4000000
    trajectory = []

    state = env.reset()
    vel = state[1]

    best_reward = -np.inf
    for i in range(transitions):
        if random.random() < eps_reducer(eps, i):
            action = env.action_space.sample()
        else:
            action = ql.act(state)

        next_state, reward, done, _ = env.step(action)
        next_state = next_state
        next_vel = next_state[1]

        reward += reward_shaping_constant * (GAMMA * abs(next_vel) - abs(vel))

        trajectory.append((state, action, next_state, reward, done))
        
        if done:
            for transition in reversed(trajectory):
                ql.update(transition)
            trajectory = []

        if not done:
            state = next_state
            vel = next_vel.copy()
        else:
            state = env.reset()
            vel = state[1]
        
        if (i + 1) % (transitions//100) == 0:
            rewards = evaluate_policy(ql, 5)
            cur_rew = np.mean(rewards)
            print(f"Step: {i+1}, Reward mean: {cur_rew}, Reward std: {np.std(rewards)}")

            if save_best:
                if cur_rew > best_reward:
                    best_reward = cur_rew
                    ql.save(os.path.join('agents', agent_name+'.npz'))
            else:
                ql.save(os.path.join('agents', agent_name+'.npz'))
