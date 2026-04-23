"""Generate all plots for the DQN vs Double DQN comparison.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = {
    "dqn": "#1f77b4",
    "ddqn": "#ff7f0e",
    "random": "#7f7f7f",
}
LABELS = {
    "dqn": "DQN",
    "ddqn": "Double DQN",
    "random": "Random",
}

ENVS = ["CartPole-v1", "LunarLander-v3"]


def smooth(data: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_training_results(env_name: str) -> dict:
    path = os.path.join("results", env_name, "training_results.json")
    with open(path) as f:
        return json.load(f)


def load_eval_results(env_name: str) -> dict:
    path = os.path.join("results", env_name, "eval_results.json")
    with open(path) as f:
        return json.load(f)


# Plot 1: Learning Curves 
def plot_learning_curves(env_name: str, results: dict, save_dir: str):
    """Plot average episodic reward vs. training episode."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_type in ["random", "dqn", "ddqn"]:
        if agent_type not in results:
            continue

        rewards_per_seed = results[agent_type]["episode_rewards"]
        min_len = min(len(r) for r in rewards_per_seed)
        rewards_array = np.array([r[:min_len] for r in rewards_per_seed])

        mean_rewards = rewards_array.mean(axis=0)
        std_rewards = rewards_array.std(axis=0)

        window = 20
        episodes = np.arange(len(mean_rewards))

        if agent_type == "random":
            ax.axhline(
                y=mean_rewards.mean(),
                color=COLORS[agent_type],
                linestyle="--",
                linewidth=1.5,
                label=f"{LABELS[agent_type]} (avg: {mean_rewards.mean():.1f})",
            )
        else:
            smoothed_mean = smooth(mean_rewards, window)
            smoothed_std = smooth(std_rewards, window)
            smoothed_eps = np.arange(len(smoothed_mean))

            ax.plot(
                smoothed_eps,
                smoothed_mean,
                color=COLORS[agent_type],
                linewidth=2,
                label=LABELS[agent_type],
            )
            ax.fill_between(
                smoothed_eps,
                smoothed_mean - smoothed_std,
                smoothed_mean + smoothed_std,
                alpha=0.2,
                color=COLORS[agent_type],
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episodic Reward")
    ax.set_title(f"Learning Curves — {env_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"learning_curves_{env_name.replace('-', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Plot 2: Q-value Overestimation 
def plot_q_values(env_name: str, results: dict, save_dir: str):
    """Plot average predicted Q-values over training for DQN vs DDQN."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for agent_type in ["dqn", "ddqn"]:
        if agent_type not in results or "q_estimates" not in results[agent_type]:
            continue

        q_estimates_per_seed = results[agent_type]["q_estimates"]

        all_episodes = set()
        for seed_data in q_estimates_per_seed:
            for ep, _ in seed_data:
                all_episodes.add(ep)
        all_episodes = sorted(all_episodes)

        q_matrix = []
        for seed_data in q_estimates_per_seed:
            ep_to_q = dict(seed_data)
            row = [ep_to_q.get(ep, np.nan) for ep in all_episodes]
            q_matrix.append(row)
        q_matrix = np.array(q_matrix)

        mean_q = np.nanmean(q_matrix, axis=0)
        std_q = np.nanstd(q_matrix, axis=0)

        ax.plot(
            all_episodes,
            mean_q,
            color=COLORS[agent_type],
            linewidth=2,
            label=LABELS[agent_type],
        )
        ax.fill_between(
            all_episodes,
            mean_q - std_q,
            mean_q + std_q,
            alpha=0.2,
            color=COLORS[agent_type],
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Max Q-value (on fixed states)")
    ax.set_title(f"Q-value Estimates During Training — {env_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"q_values_{env_name.replace('-', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# Plot 3: Evaluation Bar Chart
def plot_evaluation(env_name: str, eval_results: dict, save_dir: str):
    """Bar chart of mean evaluation scores with error bars."""
    fig, ax = plt.subplots(figsize=(8, 5))

    agents = []
    means = []
    stds = []

    for agent_type in ["random", "dqn", "ddqn"]:
        if agent_type in eval_results:
            agents.append(LABELS[agent_type])
            means.append(eval_results[agent_type]["mean"])
            stds.append(eval_results[agent_type]["std"])

    bars = ax.bar(
        agents,
        means,
        yerr=stds,
        capsize=8,
        color=[COLORS.get(a.lower().replace(" ", "").replace("double", "d"), "#999") for a in agents],
        edgecolor="black",
        linewidth=0.8,
    )

    color_map = {"Random": COLORS["random"], "DQN": COLORS["dqn"], "Double DQN": COLORS["ddqn"]}
    for bar, agent_label in zip(bars, agents):
        bar.set_facecolor(color_map.get(agent_label, "#999"))

    ax.set_ylabel("Mean Evaluation Reward")
    ax.set_title(f"Evaluation Performance (100 episodes, ε=0.05) — {env_name}")
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 2,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    path = os.path.join(save_dir, f"evaluation_{env_name.replace('-', '_')}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--env", type=str, nargs="+", default=ENVS)
    args = parser.parse_args()

    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)

    for env_name in args.env:
        print(f"\nGenerating plots for {env_name}...")

        try:
            training_results = load_training_results(env_name)
            plot_learning_curves(env_name, training_results, plots_dir)
            plot_q_values(env_name, training_results, plots_dir)
        except FileNotFoundError:
            print(f"  Training results not found for {env_name}, skipping learning curves & Q-values.")

        try:
            eval_results = load_eval_results(env_name)
            plot_evaluation(env_name, eval_results, plots_dir)
        except FileNotFoundError:
            print(f"  Eval results not found for {env_name}, skipping evaluation plot.")

    print(f"\n✓ All plots saved to {plots_dir}/")


if __name__ == "__main__":
    main()
