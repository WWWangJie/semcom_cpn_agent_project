from dataclasses import dataclass, asdict
from typing import List, Dict
import pandas as pd


@dataclass
class EpisodeResult:
    scheduler: str
    episode: int
    user_id: int
    node_id: int
    tokens: int
    snr_db: float
    fbl_error: float
    semantic_accuracy: float
    compute_delay_s: float
    tx_delay_s: float
    total_delay_s: float
    e2e_success_prob: float
    e2e_error_prob: float
    deadline_met: bool


def results_to_dataframe(results: List[EpisodeResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def summarize_results(results: List[EpisodeResult]) -> pd.DataFrame:
    df = results_to_dataframe(results)
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("scheduler")
        .agg(
            avg_e2e_success=("e2e_success_prob", "mean"),
            avg_e2e_error=("e2e_error_prob", "mean"),
            avg_fbl_error=("fbl_error", "mean"),
            avg_semantic_accuracy=("semantic_accuracy", "mean"),
            avg_tokens=("tokens", "mean"),
            avg_total_delay_ms=("total_delay_s", lambda x: 1000.0 * x.mean()),
            deadline_success_rate=("deadline_met", "mean"),
        )
        .reset_index()
        .sort_values("avg_e2e_success", ascending=False)
    )
    return summary


def print_summary(results: List[EpisodeResult]) -> None:
    summary = summarize_results(results)
    if summary.empty:
        print("No results to summarize.")
        return
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
