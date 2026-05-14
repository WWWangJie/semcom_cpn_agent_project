from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from semcom_cpn.config import SimulationConfig
from semcom_cpn.schedulers import TokenAwareAgentScheduler
from semcom_cpn.simulator import SemComCPNSimulator
from semcom_cpn.metrics import print_summary


config = SimulationConfig(num_users=4, num_edge_nodes=2, episodes=10, seed=1)
simulator = SemComCPNSimulator(config)

agent = TokenAwareAgentScheduler(
    bits_per_token=config.bits_per_token,
    blocklength=config.blocklength,
    bandwidth_hz=config.bandwidth_hz,
    semantic_alpha=config.semantic_alpha,
    semantic_acc_max=config.semantic_acc_max,
    latency_deadline_s=config.latency_deadline_s,
)

results = simulator.run(schedulers=[agent], episodes=config.episodes)
print_summary(results)
