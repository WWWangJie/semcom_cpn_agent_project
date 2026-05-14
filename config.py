from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Global simulation configuration.

    The values here are intentionally lightweight and can be changed directly
    for paper-level numerical experiments.
    """

    num_users: int = 8
    num_edge_nodes: int = 3
    episodes: int = 100
    seed: int = 42

    # Token range for each semantic task.
    min_tokens: int = 32
    max_tokens: int = 512

    # Each token is mapped to payload bits for physical-layer transmission.
    bits_per_token: int = 16

    # Finite blocklength setting.
    blocklength: int = 1200
    bandwidth_hz: float = 20e6
    noise_power_dbm: float = -94.0
    tx_power_dbm: float = 20.0

    # Semantic model.
    semantic_alpha: float = 0.012
    semantic_acc_max: float = 0.98

    # Computing model.
    # This project interprets computing as "token processing capability",
    # closer to LLM-era token/s than traditional CPU-cycle-only models.
    min_node_token_rate: float = 2.0e5
    max_node_token_rate: float = 8.0e5

    # Latency constraint in seconds.
    latency_deadline_s: float = 0.010

    # Agent scoring weights.
    weight_reliability: float = 0.45
    weight_semantic: float = 0.35
    weight_latency: float = 0.15
    weight_load: float = 0.05

    # Whether to save figures and CSV.
    save_outputs: bool = True
