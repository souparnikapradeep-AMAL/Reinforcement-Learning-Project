"""Evaluate trained DQN and DDQN agents with a near-greedy policy (ε=0.05).
"""

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import torch

from dqn_agent import DQNAgent
from utils import set_seed, get_device

ENV_CONFIGS = {
    "CartPole-v1": {"state_dim": 4, "action_dim": 2},
    "LunarLander-v3": {"state_dim": 8, "action_dim": 4},
}

NUM_EVAL_EPISODES = 100


def evaluate_agent(agent: DQNAgent, env_name: str, num_episodes: int = 100, seed: int = 9999) -> list:
    """Run the agent greedily for num_episodes and return rewards."""
    env = gym.make(env_name)
    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(state, eval_mode=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    env.close()
    return rewards


def evaluate_random(env_name: str, num_episodes: int = 100, seed: int = 9999) -> list:
    """Random baseline evaluation."""
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--env", type=str, nargs="+", default=list(ENV_CONFIGS.keys()))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    device = get_device()

    for env_name in args.env:
        config = ENV_CONFIGS[env_name]
        print(f"\n{'='*60}")
        print(f"Evaluating on: {env_name}")
        print(f"{'='*60}")

        eval_results = {}

        # Random baseline
        random_rewards = evaluate_random(env_name, NUM_EVAL_EPISODES)
        eval_results["random"] = {
            "mean": float(np.mean(random_rewards)),
            "std": float(np.std(random_rewards)),
            "rewards": random_rewards,
        }
        print(f"  Random:  {np.mean(random_rewards):.1f} ± {np.std(random_rewards):.1f}")

        # Trained agents
        for agent_type in ["dqn", "ddqn"]:
            use_double = agent_type == "ddqn"
            label = agent_type.upper()
            all_rewards = []

            for seed in args.seeds:
                model_path = os.path.join("results", env_name, agent_type, f"model_seed{seed}.pt")
                if not os.path.exists(model_path):
                    print(f"  {label} seed {seed}: model not found at {model_path}, skipping.")
                    continue

                agent = DQNAgent(
                    state_dim=config["state_dim"],
                    action_dim=config["action_dim"],
                    use_double=use_double,
                    device=device,
                )
                agent.load(model_path)
                rewards = evaluate_agent(agent, env_name, NUM_EVAL_EPISODES, seed=seed)
                all_rewards.extend(rewards)
                print(f"  {label} seed {seed}: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")

            if all_rewards:
                eval_results[agent_type] = {
                    "mean": float(np.mean(all_rewards)),
                    "std": float(np.std(all_rewards)),
                    "rewards": all_rewards,
                }
                print(f"  {label} overall: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")

        # Save evaluation results
        results_dir = os.path.join("results", env_name)
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "eval_results.json"), "w") as f:
            json.dump(eval_results, f, indent=2)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
