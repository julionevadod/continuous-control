import torch
from torch import nn
from torch.nn import functional as f


class Actor(nn.Module):
    def __init__(self, input_size: int, output_size: int, fc1_size: int = 128, fc2_size: int = 64):
        super().__init__()

        self.output_size = output_size
        self.bn0 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, fc1_size, dtype=torch.float32)
        self.bn1 = nn.BatchNorm1d(fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size, dtype=torch.float32)
        self.bn2 = nn.BatchNorm1d(fc2_size)
        self.fc3 = nn.Linear(fc2_size, output_size, dtype=torch.float32)

    def forward(self, state):
        fc1_output = f.relu(self.fc1(state))
        fc2_output = f.relu(self.fc2(fc1_output))
        fc3_output = f.tanh(self.fc3(fc2_output))
        return fc3_output
