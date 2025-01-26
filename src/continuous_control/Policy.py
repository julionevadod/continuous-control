import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, input_size: int, output_size: int, fc1_size: int = 64, fc2_size: int = 64):
        super().__init__()

        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(input_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, output_size),
        )

    def forward(self, state):
        return self.net(state)

    def act(self, state):
        output = self.forward(state).squeeze()
        means = output[torch.arange(0, self.output_size, 2)].clip(-1, 1)
        stds = output[torch.arange(1, self.output_size, 2)].clip(0)
        action = torch.normal(means, stds)
        return action
