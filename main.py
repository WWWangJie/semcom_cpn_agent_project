import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .config import SimulationConfig
from .schedulers import RandomScheduler, GreedyChannelScheduler, TokenAwareAgentScheduler
from .simulator import SemComCPNSimulator
from .metrics import results_to_dataframe, summarize_results, print_summary


def build_schedulers(config: SimulationConfig):
    return [
        RandomScheduler(),
        GreedyChannelScheduler(),
        TokenAwareAgentScheduler(
            bits_per_token=config.bits_per_token,
            blocklength=config.blocklength,
            bandwidth_hz=config.bandwidth_hz,
            semantic_alpha=config.semantic_alpha,
            semantic_acc_max=config.semantic_acc_max,
            latency_deadline_s=config.latency_deadline_s,
            weight_reliability=config.weight_reliability,
            weight_semantic=config.weight_semantic,
            weight_latency=config.weight_latency,
            weight_load=config.weight_load,
        ),
    ]


def save_outputs(results, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_dataframe(results)
    summary = summarize_results(results)

    df.to_csv(output_dir / "episode_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    # Figure 1: average e2e success.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(summary["scheduler"], summary["avg_e2e_success"])
    ax.set_ylabel("Average end-to-end success probability")
    ax.set_xlabel("Scheduler")
    ax.set_title("SemCom-CPN scheduling comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_e2e_success.png", dpi=200)
    plt.close(fig)

    # Figure 2: average delay.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(summary["scheduler"], summary["avg_total_delay_ms"])
    ax.set_ylabel("Average total delay (ms)")
    ax.set_xlabel("Scheduler")
    ax.set_title("Average end-to-end delay")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "avg_delay.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="SemCom-CPN Agent Simulation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-users", type=int, default=8)
    parser.add_argument("--num-edge-nodes", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SimulationConfig(
        episodes=args.episodes,
        seed=args.seed,
        num_users=args.num_users,
        num_edge_nodes=args.num_edge_nodes,
        save_outputs=not args.no_save,
    )

    simulator = SemComCPNSimulator(config)
    schedulers = build_schedulers(config)
    results = simulator.run(schedulers=schedulers, episodes=config.episodes)

    print_summary(results)

    if config.save_outputs:
        save_outputs(results, Path(args.output_dir))
        print(f"\nSaved CSV and figures to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
