"""Training script for DQN and Double DQN on CartPole-v1 and LunarLander-v3."""

import argparse
import json
import os
import time

import gymnasium as gym
import numpy as np

from dqn_agent import DQNAgent
from utils import set_seed, get_device

ENV_CONFIGS = {
    "CartPole-v1": {
        "num_episodes": 500,
        "state_dim": 4,
        "action_dim": 2,
    },
    "LunarLander-v3": {
        "num_episodes": 1000,
        "state_dim": 8,
        "action_dim": 4,
    },
}


Q_LOG_INTERVAL = 10

NUM_FIXED_STATES = 256


def collect_fixed_states(env_name: str, n: int = NUM_FIXED_STATES, seed: int = 999) -> np.ndarray:
    """Collect a fixed set of states by running a random policy.

    These are used throughout training to track Q-value estimates
    on a consistent set of inputs.
    """
    env = gym.make(env_name)
    states = []
    state, _ = env.reset(seed=seed)
    for _ in range(n * 2):  
        states.append(state)
        action = env.action_space.sample()
        next_state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()
    indices = np.random.choice(len(states), size=n, replace=False)
    return np.array([states[i] for i in indices], dtype=np.float32)


def run_random_baseline(env_name: str, num_episodes: int, seed: int = 42) -> list:
    """Run a random agent and return per-episode rewards."""
    env = gym.make(env_name)
    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return rewards


def train_agent(
    env_name: str,
    use_double: bool,
    seed: int,
    device: str = "cpu",
) -> dict:
    """Train a single DQN or DDQN agent and return logged metrics.

    Returns:
        dict with keys: episode_rewards, episode_losses, q_estimates, steps
    """
    config = ENV_CONFIGS[env_name]
    num_episodes = config["num_episodes"]

    set_seed(seed)
    env = gym.make(env_name)

    agent = DQNAgent(
        state_dim=config["state_dim"],
        action_dim=config["action_dim"],
        use_double=use_double,
        device=device,
    )

    # Fixed states for Q-value tracking
    fixed_states = collect_fixed_states(env_name, seed=seed)

    # Logging
    episode_rewards = []
    episode_losses = []
    q_estimates = []  # (episode, mean_max_q) pairs

    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        ep_losses = []
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action, reward, next_state, float(done))
            metrics = agent.train_step()

            if metrics:
                ep_losses.append(metrics["loss"])

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_losses.append(np.mean(ep_losses) if ep_losses else 0.0)

        # Periodically log Q-value estimates on fixed states
        if ep % Q_LOG_INTERVAL == 0:
            mean_q = agent.estimate_q_values(fixed_states)
            q_estimates.append((ep, mean_q))

        # Print progress
        if (ep + 1) % 50 == 0:
            recent_avg = np.mean(episode_rewards[-50:])
            agent_name = "DDQN" if use_double else "DQN"
            print(
                f"  [{agent_name}] {env_name} | Seed {seed} | "
                f"Ep {ep+1}/{num_episodes} | "
                f"Avg(50): {recent_avg:.1f} | ε: {agent.epsilon:.3f}"
            )

    env.close()


    agent_name = "ddqn" if use_double else "dqn"
    model_dir = os.path.join("results", env_name, agent_name)
    os.makedirs(model_dir, exist_ok=True)
    agent.save(os.path.join(model_dir, f"model_seed{seed}.pt"))

    return {
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "q_estimates": q_estimates,
    }


def main():
    parser = argparse.ArgumentParser(description="Train DQN / DDQN agents")
    parser.add_argument(
        "--env",
        type=str,
        nargs="+",
        default=list(ENV_CONFIGS.keys()),
        help="Environment(s) to train on",
    )
    parser.add_argument(
        "--agents",
        type=str,
        nargs="+",
        default=["dqn", "ddqn"],
        choices=["dqn", "ddqn"],
        help="Agent type(s) to train",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for repeated runs",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}\n")

    for env_name in args.env:
        print(f"{'='*60}")
        print(f"Environment: {env_name}")
        print(f"{'='*60}")

        all_results = {}

        #Random baseline
        print("\nRunning random baseline...")
        random_rewards = run_random_baseline(
            env_name, ENV_CONFIGS[env_name]["num_episodes"]
        )
        all_results["random"] = {"episode_rewards": [random_rewards]}
        print(f"  Random baseline avg reward: {np.mean(random_rewards):.1f}")

        #DQN / DDQN
        for agent_type in args.agents:
            use_double = agent_type == "ddqn"
            agent_label = "DDQN" if use_double else "DQN"
            print(f"\nTraining {agent_label}...")

            seed_results = {"episode_rewards": [], "episode_losses": [], "q_estimates": []}

            for seed in args.seeds:
                t0 = time.time()
                result = train_agent(env_name, use_double, seed, device)
                elapsed = time.time() - t0

                seed_results["episode_rewards"].append(result["episode_rewards"])
                seed_results["episode_losses"].append(result["episode_losses"])
                seed_results["q_estimates"].append(result["q_estimates"])

                print(f"  Seed {seed} done in {elapsed:.1f}s")

            all_results[agent_type] = seed_results

        #Save results to JSON
        results_dir = os.path.join("results", env_name)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, "training_results.json")

        # Convert numpy types for JSON serialization
        def to_serializable(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(results_path, "w") as f:
            json.dump(all_results, f, default=to_serializable, indent=2)
        print(f"\nResults saved to {results_path}")

    print("\n✓ All training complete!")


if __name__ == "__main__":
    main()
