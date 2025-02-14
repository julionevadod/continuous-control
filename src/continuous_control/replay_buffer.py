from collections import deque

import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)

    def __len__(self):
        return len(self.buffer)

    def insert(self, experience):
        """Add a new experience to the buffer

        :param experience: _description_
        :type experience: _type_
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """_summary_

        :param batch_size: _description_
        :type batch_size: _type_
        """

        sample_indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        return [self.buffer[i] for i in sample_indices]
