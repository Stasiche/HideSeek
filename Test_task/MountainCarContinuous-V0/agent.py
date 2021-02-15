from model import Model
import torch


class Agent:
    def __init__(self, device, bnh, path='agent'):
        self.model = Model(bnh).to(device)
        self.model.load_state_dict(torch.load(path))
        self.device = device
        self.bnh = bnh
        
    def act(self, state):
        return self.model(torch.tensor(state).to(self.device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()

    def reset(self):
        pass
