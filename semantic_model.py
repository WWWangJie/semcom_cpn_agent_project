import math


def semantic_accuracy_from_tokens(
    tokens: int,
    complexity: float,
    alpha: float,
    acc_max: float,
) -> float:
    """
    Nonlinear token-to-semantic-accuracy model.

    A larger token budget improves semantic accuracy, while more complex tasks
    require more tokens to reach the same accuracy.
    """
    if tokens <= 0:
        return 0.0

    effective_tokens = tokens / max(complexity, 1e-6)
    acc = acc_max * (1.0 - math.exp(-alpha * effective_tokens))
    return float(min(max(acc, 0.0), acc_max))


def semantic_recovery_success(
    semantic_accuracy: float,
    fbl_success_probability: float,
    compute_success_probability: float,
) -> float:
    """
    End-to-end semantic recovery probability.

    This is intentionally multiplicative to expose the coupling among:
    - physical-layer reliability,
    - semantic information sufficiency,
    - computing-side completion.
    """
    return float(
        min(
            max(semantic_accuracy * fbl_success_probability * compute_success_probability, 0.0),
            1.0,
        )
    )
