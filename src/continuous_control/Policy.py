from torch import nn


class Policy(nn.Module):
    def __init__(self, input_size, output_size, fc1_size, fc2_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, output_size),
        )

    def forward(self, state):
        return self.net(state)
