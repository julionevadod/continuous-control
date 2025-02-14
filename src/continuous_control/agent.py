import collections
import copy

import numpy as np
import torch
from torch import nn
from unityagents import UnityEnvironment

from .actor import Actor
from .critic import Critic
from .OUNoise import OUNoise
from .replay_buffer import ExperienceReplayBuffer

INPUT_SIZE = 33
ACTION_SIZE = 4
NUM_AGENTS = 1
MAX_LEN_EPISODE = 1000
UPDATE_EVERY = 1
WINDOW_SIZE = 100
BUFFER_SIZE = 100000
TAU = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"


class Agent:
    def __init__(self, gamma: float, lr_actor: float, lr_critic: float, save_path: str):
        """Initialize an Agent for the Reacher environment

        :param gamma: Discount rate for future rewards
        :type gamma: float
        :param lr_actor: Learning rate for the Actor
        :type lr_actor: float
        :param lr_critic: Learning rate for the Critic
        :type lr_critic: float
        :param save_path: Path to save & load network and optimizer state dicts
        :type save_path: str
        """
        torch.manual_seed(0)
        self.env = UnityEnvironment(file_name="../env/Reacher.app")
        self.actor_local = Actor(INPUT_SIZE, ACTION_SIZE).to(device).eval()
        self.actor_target = copy.deepcopy(self.actor_local).to(device).eval()
        self.critic_local = Critic(INPUT_SIZE, ACTION_SIZE, 1).to(device).eval()  # 1 since we output reward directly
        self.critic_target = copy.deepcopy(self.critic_local).to(device).eval()
        self.loss = torch.nn.functional.mse_loss
        self.replay_buffer = ExperienceReplayBuffer(BUFFER_SIZE)
        self.optimizer_actor = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.noise = OUNoise(ACTION_SIZE, 0)
        self.save_path = save_path

    def _soft_update(self, source_network: nn.Module, target_network: nn.Module):
        """Soft update from source to target nework. Parameter mix is
        defined by TAU constant

        :param source_network: network from which wegiths are copied
        :type source_network: nn.Module
        :param target_network: network to which weigths are copied
        :type target_network: nn.Module
        """
        for source_param, target_param in zip(source_network.parameters(), target_network.parameters()):
            target_param.data.copy_(TAU * source_param.data + (1 - TAU) * target_param.data)

    def _update(self, experiences: list[tuple[list[float], int, list[float], float, int]]):
        """Update policy from a batch of experiences

        :param experiences: Experiences sampled from buffer used to update local networks
        :type experiences: list[tuple[list[float],int,list[float],float,int]]
        """
        self.actor_local.train()
        self.critic_local.train()

        # Split experiences
        states, actions, next_states, rewards, dones = zip(*experiences)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device).unsqueeze(1)
        dones = torch.tensor(np.array(dones), dtype=torch.int64).to(device).unsqueeze(1)

        ########################################################################
        # Update Critic
        ########################################################################
        # Compute critic target value
        with torch.no_grad():
            actor_target_actions = self.actor_target(next_states)
            critic_target_output = self.critic_target(next_states, actor_target_actions)
        critic_target_reward = rewards + self.gamma * critic_target_output.detach() * (1 - dones)
        # Compute critic local value
        critic_local_reward = self.critic_local(states, actions)
        # Compute loss
        loss_critic = self.loss(critic_local_reward, critic_target_reward)
        # Backpropagation
        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()

        ########################################################################
        # Update Actor
        ########################################################################
        actor_local_actions = self.actor_local(states)
        critic_local_reward = self.critic_local(states, actor_local_actions)
        loss_actor = -critic_local_reward.mean()  # error I had sum instead of mean
        # Backpropagation
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()

        ########################################################################
        # Update Target Networks
        ########################################################################
        self._soft_update(self.critic_local, self.critic_target)
        self.critic_local.eval()  # TODO do we want gradients from actor to backpropagate to critic?
        self._soft_update(self.actor_local, self.actor_target)
        self.actor_local.eval()

    def save(self):
        """Save local, target and optimizers"""
        torch.save(self.actor_local.state_dict(), f"{self.save_path}/actor_local_network")
        torch.save(self.actor_target.state_dict(), f"{self.save_path}/actor_target_network")
        torch.save(self.critic_local.state_dict(), f"{self.save_path}/critic_local_network")
        torch.save(self.critic_target.state_dict(), f"{self.save_path}/critic_target_network")
        torch.save(self.optimizer_actor.state_dict(), f"{self.save_path}/actor_local_optimizer")
        torch.save(self.optimizer_critic.state_dict(), f"{self.save_path}/critic_local_optimizer")

    def load(self):
        """Load local, target and optimizers"""
        self.actor_local.load_state_dict(torch.load(f"{self.save_path}/actor_local_network", weights_only=True))
        self.actor_target.load_state_dict(torch.load(f"{self.save_path}/actor_target_network", weights_only=True))
        self.optimizer_actor.load_state_dict(torch.load(f"{self.save_path}/actor_local_optimizer", weights_only=True))
        self.critic_local.load_state_dict(torch.load(f"{self.save_path}/critic_local_network", weights_only=True))
        self.critic_target.load_state_dict(torch.load(f"{self.save_path}/critic_target_network", weights_only=True))
        self.optimizer_critic.load_state_dict(torch.load(f"{self.save_path}/critic_local_optimizer", weights_only=True))

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
                with torch.no_grad():
                    action = (
                        self.actor_local(torch.tensor(state, dtype=torch.float32).to(device)).cpu().numpy()
                        + self.noise.sample()
                    )
                action = np.clip(action, -1, 1)
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

            if float(np.mean(scores_window) > 30):
                print(f"\nEnvironment solved in {i} iterations with a score of {float(np.mean(scores_window)):.2f}")
                self.save()
                break
            if (i % 100 == 0) & (i != 0):
                print(
                    f"\rITERATION {i}/{n_iterations}: Average Reward Last 100: {float(np.mean(scores_window)):.2f} \t Last Episode: {score:.2f}"
                )
            self.noise.reset()
        return scores

    def play(self):
        """Play an episode in the Reacher environment with current Agent weights"""
        brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=False)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(NUM_AGENTS)
        while True:
            with torch.no_grad():
                action = self.actor_local(torch.tensor(states, dtype=torch.float32).to(device)).cpu().numpy()
            actions = np.clip(action, -1, 1)
            env_info = self.env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            dones = env_info.local_done
            scores += env_info.rewards
            states = next_states
            if np.any(dones):
                break
        print(f"Total score (averaged over agents) this episode: {np.mean(scores)}")
