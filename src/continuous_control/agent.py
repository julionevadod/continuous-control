import numpy as np
import torch
from unityagents import UnityEnvironment

from .policy import Policy

INPUT_SIZE = 33
ACTION_SIZE = 4
NUM_AGENTS = 1
TRAJECTORIES_SAMPLE_SIZE = 8
MAX_LEN_EPISODE = 1000


class Agent:
    def __init__(self, lr=1e-2):
        self.env = UnityEnvironment(file_name="../env/Reacher.app")
        self.policy = Policy(INPUT_SIZE, 2 * ACTION_SIZE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = 0.95  # Discounts are needed because otherwise gradients explode due to huge losses

    def _sample_trajectories(self):
        trajectories = []
        for _ in range(TRAJECTORIES_SAMPLE_SIZE):
            states_sample = []
            actions_sample = []
            log_proba_sample = []
            scores = []
            brain_name = self.env.brain_names[0]
            env_info = self.env.reset(train_mode=True)[brain_name]
            states = torch.Tensor(env_info.vector_observations)
            states_sample.append(states)
            for _ in range(MAX_LEN_EPISODE):
                actions, log_proba = self.policy.act(states)
                actions_sample.append(actions)
                log_proba_sample.append(log_proba)
                env_info = self.env.step(actions.detach().numpy())[brain_name]
                next_states = torch.Tensor(env_info.vector_observations)
                dones = env_info.local_done
                scores += env_info.rewards
                states = next_states
                states_sample.append(states)
                if np.any(dones):
                    break
            trajectories.append({
                "states": states_sample,
                "log_proba": log_proba_sample,
                "actions": actions_sample,
                "rewards": scores,
            })
        return trajectories

    def _compute_loss(self, trajectories):
        loss = []
        reward = 0
        for t in trajectories:
            loss_t = []
            discounts = torch.tensor(np.array([self.gamma**i for i in range(len(t["rewards"]))]))
            r = (torch.tensor(np.array(t["rewards"])) * discounts).sum().item()
            for t2 in t["log_proba"]:
                loss_t.append(-r * t2)
            loss.append(torch.cat(loss_t).sum())
            reward += r
        return (torch.hstack(loss) / TRAJECTORIES_SAMPLE_SIZE).sum(), reward

    def learn(self, iterations):
        for i in range(iterations):
            trajectories = self._sample_trajectories()
            loss, avg_reward = self._compute_loss(trajectories)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"\rIteration {i}/{iterations}: Average reward -> {avg_reward:2f}, Loss -> {loss:2f}", end="")
            if i % 100 == 0:
                print(f"\rIteration {i}/{iterations}: Average reward -> {avg_reward:2f}, Loss -> {loss:2f}")

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
