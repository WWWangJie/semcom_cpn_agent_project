from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from semcom_cpn.config import SimulationConfig
from semcom_cpn.main import build_schedulers, save_outputs
from semcom_cpn.simulator import SemComCPNSimulator
from semcom_cpn.metrics import print_summary


def main():
    config = SimulationConfig(
        num_users=10,
        num_edge_nodes=4,
        episodes=120,
        seed=7,
        latency_deadline_s=0.010,
    )
    simulator = SemComCPNSimulator(config)
    schedulers = build_schedulers(config)

    results = simulator.run(schedulers=schedulers, episodes=config.episodes)
    print_summary(results)
    save_outputs(results, PROJECT_ROOT / "results")


if __name__ == "__main__":
    main()
