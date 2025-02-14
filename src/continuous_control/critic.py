import torch
from torch import nn
from torch.nn import functional as f


class Critic(nn.Module):
    def __init__(self, input_size: int, action_size: int, output_size: int, fc1: int = 128, fc2: int = 64):
        """Initialize critic network for State-Value function approximation

        :param input_size: Input size of the network. Corresponds to state size
        :type input_size: int
        :param action_size: Action size of the environment
        :type action_size: int
        :param output_size: Output size of the network
        :type output_size: int
        :param fc1: Size of first dense layer, defaults to 128
        :type fc1: int, optional
        :param fc2: Size of second dense layer, defaults to 64
        :type fc2: int, optional
        """
        super().__init__()

        self.fc1 = nn.Linear(input_size, fc1, dtype=torch.float32)
        self.fc2 = nn.Linear(fc1 + action_size, fc2, dtype=torch.float32)
        self.output = nn.Linear(fc2, output_size, dtype=torch.float32)

    def forward(self, state, actions):
        fc1_output = f.relu(self.fc1(state))
        fc2_output = f.relu(self.fc2(torch.cat([fc1_output, actions], dim=1)))
        return f.relu(self.output(fc2_output))
