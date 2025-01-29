import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, input_size: int, output_size: int, fc1_size: int = 8, fc2_size: int = 9):
        super().__init__()

        self.output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(input_size, fc1_size),
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, output_size),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)

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
