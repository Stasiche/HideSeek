from torch import nn


class Model(nn.Module):
    def __init__(self, bnh):
        super(Model, self).__init__()
        self.nbh = bnh
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, bnh*2)
        )

    def forward(self, x):
        return self.model(x)
