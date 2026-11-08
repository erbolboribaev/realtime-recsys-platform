"""Statistics for comparing two experiment arms."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def compare_means(control: Sequence[float], treatment: Sequence[float]) -> dict[str, float]:
    """Two-sided z-test on the difference of means (large-sample normal approximation).

    The observations must be independent, so use one number per *user*, never one per
    request: a user's repeated requests are correlated and would overstate significance.
    """
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("each arm needs at least two observations")
    mean_c, mean_t = statistics.fmean(control), statistics.fmean(treatment)
    se = math.sqrt(
        statistics.variance(control) / len(control)
        + statistics.variance(treatment) / len(treatment)
    )
    diff = mean_t - mean_c
    z = diff / se if se > 0 else 0.0
    return {
        "n_control": float(len(control)),
        "n_treatment": float(len(treatment)),
        "mean_control": mean_c,
        "mean_treatment": mean_t,
        "diff": diff,
        "lift": diff / mean_c if mean_c > 0 else math.nan,
        "ci_low": diff - 1.96 * se,
        "ci_high": diff + 1.96 * se,
        "z": z,
        "p_value": math.erfc(abs(z) / math.sqrt(2)),
    }
