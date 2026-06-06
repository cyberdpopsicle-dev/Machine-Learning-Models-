# Day 3: Double DQN (DDQN) for Stable Reinforcement Learning 

This project implements a **Double Deep Q-Network (DDQN)** agent using **PyTorch** and **Gymnasium CartPole-v1**. DDQN improves upon standard DQN by introducing a **Target Network**, reducing overestimation bias and stabilizing training.

## Features

* Double DQN architecture
* Experience Replay Buffer
* Target Network synchronization
* Epsilon-greedy exploration
* PyTorch-based neural network
* CartPole-v1 environment support

## Installation

```bash
pip install gymnasium numpy torch
```

## Architecture

* **Policy Network (θ):** Selects actions and learns from experience.
* **Target Network (θ⁻):** Provides stable target Q-values and updates periodically.
* **Replay Buffer:** Stores past transitions for randomized training.

## Training

Run the training script:

```bash
python train_ddqn.py
```

The agent learns to balance the CartPole by combining replay memory, target-network updates, and Double DQN target estimation.

## Key DDQN Update

Instead of using one network for both action selection and evaluation:

1. Policy Network selects the best next action.
2. Target Network evaluates that action.
3. Target Network weights are synchronized every few episodes.

This reduces maximization bias and improves learning stability.

## Expected Results

Compared to a standard DQN, DDQN typically delivers:

* Smoother training curves
* More stable convergence
* Reduced Q-value overestimation
* Faster achievement of high CartPole scores

## Tech Stack

* Python
* PyTorch
* Gymnasium
* NumPy

## License

Educational and research use.

## Architecture

```text
┌──────────────────────────────────────────────┐
│          Select Max Action (Best a')         │
▼                                              │
┌───────────────────┐               ┌──────────┴──────────┐
│                   │               │                     │
│   Policy Network  ├──────────────►│    Target Network   │
│    Weights (θ)    │               │    Weights (θ⁻)     │
│                   │               │                     │
└───────────────────┘               └──────────┬──────────┘
         ▲                                     │
         │                                     ▼
         └──── Updates θ⁻ Every N Steps ──────┘
```

* **Policy Network (θ):** Selects actions and learns from experience.
* **Target Network (θ⁻):** Provides stable target Q-values and updates periodically.
* **Replay Buffer:** Stores past transitions for randomized training.

The Policy Network chooses the best next action, while the Target Network evaluates that action using frozen weights. Periodic synchronization keeps learning stable and reduces Q-value overestimation.

