import numpy as np
import torch
from torch import nn
from torch.nn import functional as f


class Policy(nn.Module):
    def __init__(self, input_size: int, output_size: int, fc1_size: int = 8, fc2_size: int = 9):
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
        x = self.net(state)

        output = torch.zeros_like(x)

        for i in range(0, output.shape[1], 2):
            output[:, i] = f.tanh(x[:, i])
            output[:, i + 1] = f.sigmoid(x[:, i + 1])  # let's try to use relu

        return output

    def get_log_proba(self, action, mean, std):
        log_proba = -((action - mean) ** 2) / (2 * (std**2)) - torch.log(np.sqrt(2 * torch.pi) * std)
        return log_proba  # Cannot use nan to num because it is not differentiable

    def act(self, state):
        output = self.forward(state).squeeze()
        means = output[torch.arange(0, self.output_size, 2)]  # .clip(min=-1, max=1)
        stds = output[torch.arange(1, self.output_size, 2)]  # .clip(min=0)
        m = torch.distributions.Normal(means, stds)
        action = m.sample()
        log_proba = m.log_prob(action)
        # action = torch.normal(means, stds)
        # log_proba = self.get_log_proba(action, means, stds) #log proba results is the same as sampling from torch.distributions
        return action, log_proba
