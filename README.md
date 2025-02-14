# banana-collector-rl

Current projects trains a Actor-Critic architecture to solve Reacher unity environment.

## Project Details

### The Environment

States are defined in a 3e-dimensional space. Action space size is 4, being each of 4 actions a float number between -1 and 1.

### The task

The task is episodic, meaning that it has a defined end state (marked by done flag coming from environment). The task is considered to be solved when the agent achieves an average reward of +13 over 100 consecutive episodes.

## Getting started

. Fork the `continuous-control` repo on GitHub.

1. Clone your fork locally:

```bash
cd <directory_in_which_repo_should_be_created>
git clone git@github.com:YOUR_NAME/continuous-control.git
```

2. Now we need to install the environment. Navigate into the directory

```bash
cd continuous-control
```

3. Then, install the environment with:

```bash
uv sync
```

4. Activate the environment (from banana-collector-rl folder):

```bash
source ../.venv/bin/activate
```

5. Place Reacher.app from course resources inside **continuous-control**.

## Instructions

Once environment has been activated, training can be run from command line:

```bash
python -m train
```

Running the module as it is will run the training loop with default parameters. These default parameters produce an agent that solves the environment. However, agent hyperparameters can be configured by means of runtime arguments:

#### `-i`, `--iterations`

- **Description**: Number of environment steps to train for.
- **Default**: Value from `config["DEFAULT"]["ITERATIONS"]`.

#### `-b`, `--batch_size`

- **Description**: Batch size for learning steps.
- **Default**: Value from `config["DEFAULT"]["BATCH_SIZE"]`.

#### `-g`, `--gamma`

- **Description**: Discount rate used in reinforcement learning.
- **Default**: Value from `config["DEFAULT"]["GAMMA"]`.

#### `-a`, `--learning_rate_actor`

- **Description**: Learning rate for the Actor optimizer.
- **Default**: Value from `config["DEFAULT"]["LEARNING_RATE_ACTOR"]`.

#### `-c`, `--learning_rate_critic`

- **Description**: Learning rate for the Critic optimizer.
- **Default**: Value from `config["DEFAULT"]["LEARNING_RATE_CRITIC"]`.
