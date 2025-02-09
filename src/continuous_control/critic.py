import torch
from torch import nn
from torch.nn import functional as f


class Critic(nn.Module):
    def __init__(self, input_size, action_size, output_size, fc1: int = 400, fc2: int = 300):
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1, dtype=torch.float32)
        self.fc2 = nn.Linear(fc1 + action_size, fc2, dtype=torch.float32)
        self.output = nn.Linear(fc2, output_size, dtype=torch.float32)

    def forward(self, state, actions):
        fc1_output = f.relu(self.fc1(state))
        fc2_output = f.relu(self.fc2(torch.cat([fc1_output, actions], dim=1)))
        return f.relu(self.output(fc2_output))
