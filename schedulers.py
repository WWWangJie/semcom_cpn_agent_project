from abc import ABC, abstractmethod
from typing import List
import numpy as np

from .entities import SemanticTask, EdgeNode, ChannelState, ScheduleDecision
from .fbl import fbl_error_probability, db_to_linear, awgn_capacity_bits, transmission_delay
from .semantic_model import semantic_accuracy_from_tokens


class BaseScheduler(ABC):
    name: str = "base"

    @abstractmethod
    def decide(
        self,
        task: SemanticTask,
        edge_nodes: List[EdgeNode],
        channel_state: ChannelState,
        rng: np.random.Generator,
    ) -> ScheduleDecision:
        raise NotImplementedError


class RandomScheduler(BaseScheduler):
    name = "random"

    def decide(
        self,
        task: SemanticTask,
        edge_nodes: List[EdgeNode],
        channel_state: ChannelState,
        rng: np.random.Generator,
    ) -> ScheduleDecision:
        node = rng.choice(edge_nodes)
        tokens = int(rng.integers(task.min_tokens, task.max_tokens + 1))
        return ScheduleDecision(task.user_id, node.node_id, tokens)


class GreedyChannelScheduler(BaseScheduler):
    """
    Baseline: select the edge node with the best instantaneous SNR and use
    a medium-to-high token budget.
    """

    name = "greedy_channel"

    def decide(
        self,
        task: SemanticTask,
        edge_nodes: List[EdgeNode],
        channel_state: ChannelState,
        rng: np.random.Generator,
    ) -> ScheduleDecision:
        best_node = max(edge_nodes, key=lambda n: channel_state.snr_db(task.user_id, n.node_id))
        tokens = int(0.75 * task.max_tokens + 0.25 * task.min_tokens)
        return ScheduleDecision(task.user_id, best_node.node_id, tokens)


class TokenAwareAgentScheduler(BaseScheduler):
    """
    Lightweight AI-Agent-like scheduler.

    This is not an LLM itself. It is a transparent rule-based Agent prototype:
    the Agent evaluates candidate (node, token) actions using a score that
    jointly considers FBL reliability, semantic accuracy, delay, and node load.

    This can be replaced by:
    - reinforcement learning,
    - LLM planning + tool execution,
    - Bayesian optimization,
    - differentiable resource allocation.
    """

    name = "token_aware_agent"

    def __init__(
        self,
        bits_per_token: int,
        blocklength: int,
        bandwidth_hz: float,
        semantic_alpha: float,
        semantic_acc_max: float,
        latency_deadline_s: float,
        weight_reliability: float = 0.45,
        weight_semantic: float = 0.35,
        weight_latency: float = 0.15,
        weight_load: float = 0.05,
        token_candidates: int = 9,
    ) -> None:
        self.bits_per_token = bits_per_token
        self.blocklength = blocklength
        self.bandwidth_hz = bandwidth_hz
        self.semantic_alpha = semantic_alpha
        self.semantic_acc_max = semantic_acc_max
        self.latency_deadline_s = latency_deadline_s
        self.weight_reliability = weight_reliability
        self.weight_semantic = weight_semantic
        self.weight_latency = weight_latency
        self.weight_load = weight_load
        self.token_candidates = token_candidates

    def decide(
        self,
        task: SemanticTask,
        edge_nodes: List[EdgeNode],
        channel_state: ChannelState,
        rng: np.random.Generator,
    ) -> ScheduleDecision:
        candidates = np.linspace(
            task.min_tokens,
            task.max_tokens,
            num=self.token_candidates,
            dtype=int,
        )
        candidates = sorted(set(int(x) for x in candidates))

        best_score = -1e18
        best_decision = None

        for node in edge_nodes:
            snr_db = channel_state.snr_db(task.user_id, node.node_id)
            snr_linear = db_to_linear(snr_db)
            spectral_eff = max(0.1, 0.65 * awgn_capacity_bits(snr_linear))

            for tokens in candidates:
                payload_bits = tokens * self.bits_per_token
                fbl_error = fbl_error_probability(
                    payload_bits=payload_bits,
                    blocklength=self.blocklength,
                    snr_db=snr_db,
                )
                phy_success = 1.0 - fbl_error
                sem_acc = semantic_accuracy_from_tokens(
                    tokens=tokens,
                    complexity=task.complexity,
                    alpha=self.semantic_alpha,
                    acc_max=self.semantic_acc_max,
                )
                comp_delay = node.processing_delay(tokens, task.complexity)
                tx_delay = transmission_delay(payload_bits, spectral_eff, self.bandwidth_hz)
                total_delay = comp_delay + tx_delay

                delay_score = max(0.0, 1.0 - total_delay / max(self.latency_deadline_s, 1e-9))
                load_score = 1.0 - node.load

                score = (
                    self.weight_reliability * phy_success
                    + self.weight_semantic * sem_acc
                    + self.weight_latency * delay_score
                    + self.weight_load * load_score
                )

                # Urgent tasks are more sensitive to deadline violation.
                if total_delay > self.latency_deadline_s:
                    score -= 0.25 * task.urgency

                if score > best_score:
                    best_score = score
                    best_decision = ScheduleDecision(
                        user_id=task.user_id,
                        node_id=node.node_id,
                        tokens=tokens,
                        metadata={
                            "score": float(score),
                            "predicted_phy_success": float(phy_success),
                            "predicted_semantic_accuracy": float(sem_acc),
                            "predicted_delay_s": float(total_delay),
                            "node_load": float(node.load),
                        },
                    )

        if best_decision is None:
            raise RuntimeError("No valid scheduling decision found.")

        return best_decision
