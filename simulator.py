from typing import List, Sequence, Optional
import numpy as np

from .config import SimulationConfig
from .entities import UserTerminal, EdgeNode, ChannelState
from .schedulers import BaseScheduler
from .fbl import fbl_error_probability, db_to_linear, awgn_capacity_bits, transmission_delay
from .semantic_model import semantic_accuracy_from_tokens, semantic_recovery_success
from .metrics import EpisodeResult


class SemComCPNSimulator:
    """
    End-to-end simulation loop for SemCom-CPN co-design.

    The simulator generates:
    - semantic tasks,
    - channel states,
    - token scheduling decisions,
    - FBL transmission reliability,
    - computing delay,
    - semantic recovery probability.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.users = [UserTerminal(user_id=i) for i in range(config.num_users)]
        self.edge_nodes = self._create_edge_nodes()

    def _create_edge_nodes(self) -> List[EdgeNode]:
        nodes = []
        for node_id in range(self.config.num_edge_nodes):
            token_rate = float(
                self.rng.uniform(
                    self.config.min_node_token_rate,
                    self.config.max_node_token_rate,
                )
            )
            nodes.append(EdgeNode(node_id=node_id, token_rate=token_rate))
        return nodes

    def _reset_loads(self) -> None:
        for node in self.edge_nodes:
            node.load = 0.0

    def _generate_channel_state(self) -> ChannelState:
        """
        Generate a random user-node SNR matrix.

        A simple model is used here:
        base SNR + shadowing + small random fluctuation.
        """
        base = self.rng.uniform(5.0, 22.0, size=(self.config.num_users, self.config.num_edge_nodes))
        fluctuation = self.rng.normal(0.0, 2.0, size=base.shape)
        snr_db = np.clip(base + fluctuation, -2.0, 30.0)
        return ChannelState(snr_db_matrix=snr_db)

    def run(self, schedulers: Sequence[BaseScheduler], episodes: Optional[int] = None) -> List[EpisodeResult]:
        episodes = int(episodes if episodes is not None else self.config.episodes)
        all_results: List[EpisodeResult] = []

        for scheduler in schedulers:
            self._reset_loads()
            for ep in range(episodes):
                channel_state = self._generate_channel_state()

                for user in self.users:
                    task = user.generate_task(
                        rng=self.rng,
                        min_tokens=self.config.min_tokens,
                        max_tokens=self.config.max_tokens,
                    )
                    decision = scheduler.decide(
                        task=task,
                        edge_nodes=self.edge_nodes,
                        channel_state=channel_state,
                        rng=self.rng,
                    )

                    node = self.edge_nodes[decision.node_id]
                    snr_db = channel_state.snr_db(task.user_id, decision.node_id)
                    payload_bits = decision.tokens * self.config.bits_per_token

                    fbl_error = fbl_error_probability(
                        payload_bits=payload_bits,
                        blocklength=self.config.blocklength,
                        snr_db=snr_db,
                    )
                    phy_success = 1.0 - fbl_error

                    sem_acc = semantic_accuracy_from_tokens(
                        tokens=decision.tokens,
                        complexity=task.complexity,
                        alpha=self.config.semantic_alpha,
                        acc_max=self.config.semantic_acc_max,
                    )

                    snr_linear = db_to_linear(snr_db)
                    spectral_eff = max(0.1, 0.65 * awgn_capacity_bits(snr_linear))
                    tx_delay = transmission_delay(payload_bits, spectral_eff, self.config.bandwidth_hz)
                    comp_delay = node.processing_delay(decision.tokens, task.complexity)
                    total_delay = tx_delay + comp_delay

                    compute_success = 1.0 if total_delay <= self.config.latency_deadline_s else max(
                        0.0,
                        1.0 - (total_delay - self.config.latency_deadline_s) / self.config.latency_deadline_s,
                    )

                    e2e_success = semantic_recovery_success(
                        semantic_accuracy=sem_acc,
                        fbl_success_probability=phy_success,
                        compute_success_probability=compute_success,
                    )
                    e2e_error = 1.0 - e2e_success
                    deadline_met = total_delay <= self.config.latency_deadline_s

                    node.update_load(decision.tokens)

                    all_results.append(
                        EpisodeResult(
                            scheduler=scheduler.name,
                            episode=ep,
                            user_id=task.user_id,
                            node_id=decision.node_id,
                            tokens=decision.tokens,
                            snr_db=snr_db,
                            fbl_error=fbl_error,
                            semantic_accuracy=sem_acc,
                            compute_delay_s=comp_delay,
                            tx_delay_s=tx_delay,
                            total_delay_s=total_delay,
                            e2e_success_prob=e2e_success,
                            e2e_error_prob=e2e_error,
                            deadline_met=deadline_met,
                        )
                    )

        return all_results
