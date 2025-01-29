import collections

import numpy as np
import torch
from unityagents import UnityEnvironment

from .actor import Actor
from .replay_buffer import ExperienceReplayBuffer

INPUT_SIZE = 33
ACTION_SIZE = 4
NUM_AGENTS = 1
TRAJECTORIES_SAMPLE_SIZE = 8
MAX_LEN_EPISODE = 1000
UPDATE_EVERY = 4
WINDOW_SIZE = 100
BUFFER_SIZE = 100000

device = "cpu"


class Agent:
    def __init__(self, lr=1e-2, gamma=0.95):
        self.env = UnityEnvironment(file_name="../env/Reacher.app")
        self.actor_local = Actor(INPUT_SIZE, ACTION_SIZE).eval()
        self.actor_target = Actor(INPUT_SIZE, ACTION_SIZE).eval()
        self.critic_local = Actor(INPUT_SIZE, ACTION_SIZE).eval()
        self.critic_target = Actor(INPUT_SIZE, ACTION_SIZE).eval()
        self.loss = torch.nn.functional.mse_loss
        self.replay_buffer = ExperienceReplayBuffer(BUFFER_SIZE)
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(), lr=lr)
        self.gamma = gamma  # Discounts are needed because otherwise gradients explode due to huge losses

    def _update(self, experiences: list[tuple[list[float], int, list[float], float, int]]):
        """Update policy from a batch of experiences

        :param experiences: Experiences sampled from buffer used to update local network
        :type experiences: list[tuple[list[float],int,list[float],float,int]]
        """
        self.actor_local.train()
        self.critic_local.train()

        # Split experiences
        states, actions, next_states, rewards, dones = zip(*experiences)
        states = torch.tensor(np.array(states), dtype=torch.float64).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float64).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float64).to(device).unsqueeze(1)
        dones = torch.tensor(np.array(dones), dtype=torch.int64).to(device).unsqueeze(1)

        # Local network pass
        local_network_output = self.critic_local(states)
        local_expected_reward = local_network_output.gather(1, actions)

        # Target network pass
        with torch.no_grad():
            target_network_output = self.critic_target(next_states)

        target_expected_reward = rewards + self.gamma * target_network_output.detach().max(dim=1).values.unsqueeze(
            1
        ) * (1 - dones)

        # Compute loss
        loss = self.loss(local_expected_reward, target_expected_reward)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._soft_update(self.local_network, self.target_network, 1e-3)
        # self.target_network = copy.deepcopy(self.local_network)
        self.eps = max(self.eps * self.eps_decay, self.eps_end)
        self.critic_local.eval()

    def learn(self, n_iterations: int, batch_size: int = 4) -> list[float]:
        """Make agent learn how to interact with its given environment

        :param n_iterations: Number of iterations to learn for
        :type n_iterations: int
        :param batch_size: Batch size used for sampling from experience replay buffer, defaults to 4
        :type batch_size: int, optional
        :return: Scores for each episode played
        :rtype: list[float]
        """
        scores = []
        scores_window = collections.deque(maxlen=WINDOW_SIZE)
        brain_name = self.env.brain_names[0]
        for i in range(n_iterations):
            env_info = self.env.reset(train_mode=True)[brain_name]
            state = env_info.vector_observations[0]
            score = 0
            for j in range(MAX_LEN_EPISODE):
                action = self.actor_target(state).detach().cpu().numpy()
                env_info = self.env.step(action)[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                score += reward
                self.replay_buffer.insert([state, action, next_state, reward, done])
                if (j % UPDATE_EVERY == 0) & (len(self.replay_buffer) >= batch_size):
                    experiences = self.replay_buffer.sample(batch_size)
                    self._update(experiences)
                if done:
                    break
                state = next_state
            scores.append(score)
            scores_window.append(score)
            print(
                f"\rITERATION {i}/{n_iterations}: Average Reward Last 100: {float(np.mean(scores_window)):.2f} \t Last Episode: {score:.2f}",
                end="",
            )
            if i % 100 == 0:
                print(
                    f"\rITERATION {i}/{n_iterations}: Average Reward Last 100: {float(np.mean(scores_window)):.2f} \t Last Episode: {score:.2f}"
                )
        return scores

    def play(self):
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(NUM_AGENTS)
        while True:
            with torch.no_grad():
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
