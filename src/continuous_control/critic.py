from torch import nn


class Critic(nn.Module):
    def __init__(self, input_size, output_size, fc1: int = 64, fc2: int = 64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, fc1), nn.ReLU(), nn.Linear(fc1, fc2), nn.ReLU(), nn.Linear(fc2, output_size)
        )

    def forward(self, state):
        return self.net(state)
