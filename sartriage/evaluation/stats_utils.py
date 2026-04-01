"""
evaluation/stats_utils.py
==========================
Statistical comparison utilities for dissertation-quality reporting.

Every comparison includes:
    - Mean ± 95% CI
    - Paired t-test (or McNemar's where appropriate)
    - Cohen's d effect size
    - Bonferroni correction for multiple comparisons

Usage:
    from evaluation.stats_utils import compare, compare_multi, bootstrap_ci
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


# ════════════════════════════════════════════════════════════════════════
# Comparison result container
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Result of a paired statistical comparison."""
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_corrected: Optional[float]  # Bonferroni-corrected
    cohens_d: float
    n_samples: int
    significant: bool
    significance_stars: str

    def to_dict(self) -> Dict:
        return {
            "name_a": self.name_a,
            "name_b": self.name_b,
            "mean_a": round(self.mean_a, 6),
            "mean_b": round(self.mean_b, 6),
            "mean_diff": round(self.mean_diff, 6),
            "ci_95": [round(self.ci_low, 6), round(self.ci_high, 6)],
            "p_value": round(self.p_value, 6),
            "p_corrected": round(self.p_corrected, 6) if self.p_corrected is not None else None,
            "cohens_d": round(self.cohens_d, 4),
            "n_samples": self.n_samples,
            "significant": self.significant,
            "stars": self.significance_stars,
        }

    def __str__(self) -> str:
        p_str = f"p={self.p_value:.4f}"
        if self.p_corrected is not None:
            p_str += f" (corr: {self.p_corrected:.4f})"
        return (
            f"{self.name_a} vs {self.name_b}: "
            f"Δ={self.mean_diff:+.4f} "
            f"CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"{p_str}, d={self.cohens_d:.2f} {self.significance_stars}"
        )


def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ════════════════════════════════════════════════════════════════════════
# Core comparison function
# ════════════════════════════════════════════════════════════════════════

def compare(
    a: Sequence[float],
    b: Sequence[float],
    name_a: str = "A",
    name_b: str = "B",
    n_comparisons: int = 1,
    alpha: float = 0.05,
    verbose: bool = True,
) -> ComparisonResult:
    """Standard paired comparison with CI, p-value, and effect size.

    Parameters
    ----------
    a, b : array-like of matched observations (e.g. fold accuracies).
    name_a, name_b : labels for the two conditions.
    n_comparisons : total number of comparisons being made (for Bonferroni).
    alpha : significance level (default 0.05).
    verbose : if True, print the result.

    Returns
    -------
    ComparisonResult
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert len(a) == len(b), f"Arrays must be same length: {len(a)} vs {len(b)}"

    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 1e-8

    # Paired t-test
    if n > 1 and std_diff > 1e-12:
        t_stat, p_val = stats.ttest_rel(a, b)
        ci = stats.t.interval(1 - alpha, n - 1,
                              loc=mean_diff,
                              scale=stats.sem(diff))
    else:
        t_stat = 0.0
        p_val = 1.0
        ci = (mean_diff, mean_diff)

    # Cohen's d
    d = mean_diff / std_diff if std_diff > 1e-12 else 0.0

    # Bonferroni correction
    p_corrected = min(p_val * n_comparisons, 1.0) if n_comparisons > 1 else None
    p_effective = p_corrected if p_corrected is not None else p_val

    result = ComparisonResult(
        name_a=name_a,
        name_b=name_b,
        mean_a=float(np.mean(a)),
        mean_b=float(np.mean(b)),
        mean_diff=mean_diff,
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        p_value=float(p_val),
        p_corrected=float(p_corrected) if p_corrected is not None else None,
        cohens_d=float(d),
        n_samples=n,
        significant=p_effective < alpha,
        significance_stars=_sig_stars(p_effective),
    )

    if verbose:
        print(f"    {result}")

    return result


def compare_multi(
    baseline: Sequence[float],
    conditions: Dict[str, Sequence[float]],
    baseline_name: str = "Full",
    alpha: float = 0.05,
    verbose: bool = True,
) -> Dict[str, ComparisonResult]:
    """Compare multiple conditions against a single baseline.

    Automatically applies Bonferroni correction for the number of conditions.

    Parameters
    ----------
    baseline : matched observations for the baseline condition.
    conditions : {name: observations} for each ablation.
    baseline_name : label for the baseline.
    alpha : significance level.

    Returns
    -------
    dict of ComparisonResult keyed by condition name.
    """
    n_comp = len(conditions)
    results = {}

    if verbose:
        print(f"\n    {'Condition':<20} {'Δ':>8} {'CI_lo':>8} {'CI_hi':>8} "
              f"{'p':>8} {'p_corr':>8} {'d':>6} {'sig':>4}")
        print(f"    {'─' * 80}")

    for name, obs in conditions.items():
        r = compare(
            baseline, obs, baseline_name, name,
            n_comparisons=n_comp, alpha=alpha, verbose=False,
        )
        results[name] = r

        if verbose:
            p_corr_str = f"{r.p_corrected:.4f}" if r.p_corrected is not None else "—"
            print(f"    {name:<20} {r.mean_diff:>+8.4f} {r.ci_low:>8.4f} "
                  f"{r.ci_high:>8.4f} {r.p_value:>8.4f} {p_corr_str:>8} "
                  f"{r.cohens_d:>6.2f} {r.significance_stars:>4}")

    return results


# ════════════════════════════════════════════════════════════════════════
# Bootstrap confidence interval
# ════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    data: Sequence[float],
    statistic: str = "mean",
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute a bootstrap confidence interval.

    Parameters
    ----------
    data : 1-D array of observations.
    statistic : one of 'mean', 'median'.
    n_bootstrap : number of bootstrap resamples.
    alpha : significance level.
    seed : random seed.

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    data = np.asarray(data, dtype=float)
    rng = np.random.default_rng(seed)

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(data))

    boot_stats = np.array([
        float(stat_fn(rng.choice(data, size=len(data), replace=True)))
        for _ in range(n_bootstrap)
    ])

    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return point, lo, hi


# ════════════════════════════════════════════════════════════════════════
# McNemar's test (for paired binary predictions)
# ════════════════════════════════════════════════════════════════════════

def mcnemar_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    name_a: str = "A",
    name_b: str = "B",
    verbose: bool = True,
) -> Dict:
    """McNemar's test for paired binary classification outcomes.

    Parameters
    ----------
    correct_a : boolean array — True where model A was correct.
    correct_b : boolean array — True where model B was correct.

    Returns
    -------
    dict with contingency table and test results.
    """
    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    assert len(correct_a) == len(correct_b)

    # 2x2 contingency: (A_right & B_right), (A_right & B_wrong), etc.
    n_both_right = int(np.sum(correct_a & correct_b))
    n_a_only = int(np.sum(correct_a & ~correct_b))
    n_b_only = int(np.sum(~correct_a & correct_b))
    n_both_wrong = int(np.sum(~correct_a & ~correct_b))

    # McNemar's with continuity correction
    n_discordant = n_a_only + n_b_only
    if n_discordant > 0:
        chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / n_discordant
        p_val = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_val = 1.0

    result = {
        "name_a": name_a,
        "name_b": name_b,
        "acc_a": round((n_both_right + n_a_only) / len(correct_a), 4),
        "acc_b": round((n_both_right + n_b_only) / len(correct_b), 4),
        "contingency": {
            "both_correct": n_both_right,
            "a_only_correct": n_a_only,
            "b_only_correct": n_b_only,
            "both_wrong": n_both_wrong,
        },
        "chi2": round(chi2, 4),
        "p_value": round(p_val, 6),
        "significant": p_val < 0.05,
        "stars": _sig_stars(p_val),
        "n_samples": len(correct_a),
    }

    if verbose:
        print(f"    McNemar {name_a} vs {name_b}: "
              f"acc_a={result['acc_a']:.1%}, acc_b={result['acc_b']:.1%}, "
              f"χ²={chi2:.2f}, p={p_val:.4f} {result['stars']}")

    return result


# ════════════════════════════════════════════════════════════════════════
# Multi-seed runner
# ════════════════════════════════════════════════════════════════════════

def multi_seed_eval(
    eval_fn,
    seeds: Sequence[int] = (42, 123, 456, 789, 1024),
    verbose: bool = True,
) -> Tuple[List[float], float, float, float]:
    """Run an evaluation function across multiple seeds.

    Parameters
    ----------
    eval_fn : callable(seed) → float (accuracy or metric value).
    seeds : seeds to use.

    Returns
    -------
    (results, mean, std, ci_95)
    """
    results = [eval_fn(s) for s in seeds]
    mean = float(np.mean(results))
    std = float(np.std(results, ddof=1))
    ci = 1.96 * std / math.sqrt(len(results))

    if verbose:
        print(f"    {len(seeds)} seeds: {mean:.4f} ± {ci:.4f} "
              f"(std={std:.4f}, seeds={list(seeds)})")

    return results, mean, std, ci
