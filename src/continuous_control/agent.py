import numpy as np
import torch
from unityagents import UnityEnvironment

from .policy import Policy

INPUT_SIZE = 33
ACTION_SIZE = 4
NUM_AGENTS = 1


class Agent:
    def __init__(self):
        self.env = UnityEnvironment(file_name="../env/Reacher.app")
        self.policy = Policy(INPUT_SIZE, ACTION_SIZE)

    def learn(self):
        pass

    def play(self):
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[brain_name]

        states = env_info.vector_observations
        scores = np.zeros(NUM_AGENTS)
        while True:
            actions = self.policy(torch.Tensor(states))
            actions = np.clip(actions.detach().numpy(), -1, 1)
            env_info = self.env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            if np.any(dones):
                break
        print(f"Total score (averaged over agents) this episode: {np.mean(scores)}")
