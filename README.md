# DQN vs Double DQN: Classic Control Comparison

**CS 5180 — Reinforcement Learning**

This project compares Deep Q-Networks (DQN) and Double DQN (DDQN) on two Gymnasium
classic control environments: CartPole-v1 and LunarLander-v3. The focus is on
demonstrating the Q-value overestimation bias in DQN and how DDQN mitigates it.

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Experiments

### 1. Train all agents
```bash
python train.py
```

This trains DQN and DDQN (3 seeds each) on both CartPole-v1 (500 episodes) and
LunarLander-v3 (1000 episodes), plus a random baseline. Results are saved to `results/`.

You can also run selectively:
```bash
python train.py --env CartPole-v1                  # Single environment
python train.py --agents ddqn                       # Only Double DQN
python train.py --seeds 0 1 2 3 4                   # 5 seeds
```

### 2. Evaluate trained agents
```bash
python evaluate.py
```

Runs each trained agent for 100 episodes with ε=0.05 and reports mean ± std scores.

### 3. Generate plots
```bash
python plot_results.py
```

Produces three plots per environment in `results/plots/`:
- **Learning curves**: Smoothed reward vs. episode with shaded std
- **Q-value analysis**: Mean predicted Q-values over training (DQN vs DDQN)
- **Evaluation bar chart**: Final performance comparison

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── replay_buffer.py       # Experience replay buffer
├── q_network.py           # Shared neural network architecture
├── dqn_agent.py           # DQN/DDQN agent (single class, `use_double` flag)
├── train.py               # Main training script
├── evaluate.py            # Post-training greedy evaluation
├── plot_results.py        # Generates all comparison plots
└── utils.py               # Seeding and device utilities
```

## Key Design Decision

DQN and DDQN share a single `DQNAgent` class. The **only** difference is in the
target computation inside `train_step()`:

```python
if self.use_double:
    # DDQN: online net selects, target net evaluates
    best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
    next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
else:
    # DQN: target net selects and evaluates
    next_q = self.target_net(next_states).max(dim=1)[0]
```

This ensures the comparison is completely fair — identical architecture, hyperparameters,
replay buffer, and epsilon schedule.

## Hyperparameters

| Parameter                    | Value                     |
|------------------------------|---------------------------|
| Learning rate                | 1e-3 (Adam)               |
| Replay buffer size           | 50,000                    |
| Batch size                   | 64                        |
| Target network update freq   | Every 500 steps           |
| Epsilon decay                | 1.0 → 0.01 over 10K steps |
| Discount factor (γ)          | 0.99                      |
| Network                      | 2 × 128 FC + ReLU         |
| Training episodes            | 500 (CartPole), 1000 (LunarLander) |
