import torch
import torch.nn.functional as F
import copy
from model import Model


class DQN:
    def __init__(self, device, gamma, bnh):
        self.bnh = bnh
        self.model = Model(bnh).to(device)
        self.target_model = copy.deepcopy(self.model)

        self.model.to(device)
        self.target_model.to(device)

        self.device = device
        self.gamma = gamma

    def __call__(self, state):
        return self.model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def act(self, state):
        return self(state)

    def update(self, batch, optimizer):
        state, action, reward, next_state, done = batch
        state = torch.tensor(state).to(self.device).float()
        next_state = torch.tensor(next_state).to(self.device).float()
        reward = torch.tensor(reward).to(self.device).float()
        action = torch.tensor(action).to(self.device)
        done = torch.tensor(done).to(self.device, int)

        with torch.no_grad():
            target_q = self.target_model(next_state).max(1)[0].view(-1)

        target_q = reward + target_q * self.gamma * (1 - done)
        q = self.model(state).gather(1, action.unsqueeze(1))
        loss = F.mse_loss(q, target_q.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    def save(self, path='agent'):
        torch.save(self.model.state_dict(), path)