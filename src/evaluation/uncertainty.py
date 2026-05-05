"""
src/evaluation/uncertainty.py
-----------------------------
Uncertainty quantification and stratified evaluation utilities.

Key conventions:
  - Entropy is used for uncertainty stratification.
  - uncertain_rate is reported separately using confidence < 0.70
    to preserve compatibility with prior result tables.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ENTROPY_LOW_THRESHOLD = 0.30
ENTROPY_HIGH_THRESHOLD = 0.65
ENTROPY_MAX_3CLASS = 1.0986  # ln(3)

CONF_UNCERTAIN_THRESHOLD = 0.70


def compute_confidence(probs: np.ndarray) -> np.ndarray:
    """Return max softmax confidence for each sample."""
    return probs.max(axis=-1)


def compute_entropy(probs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Compute Shannon entropy for each sample.

    H(x) = -sum_c p_c log(p_c)
    """
    return -(probs * np.log(probs + eps)).sum(axis=-1)


def compute_uncertainty_margin(probs: np.ndarray) -> np.ndarray:
    """
    Proxy uncertainty score based on top-2 probability margin.

    Returns:
        1 - (top1 - top2)
    """
    sorted_probs = np.sort(probs, axis=-1)[:, ::-1]
    return 1.0 - (sorted_probs[:, 0] - sorted_probs[:, 1])


def assign_strata(
    entropy: np.ndarray,
    low_threshold: float = ENTROPY_LOW_THRESHOLD,
    high_threshold: float = ENTROPY_HIGH_THRESHOLD,
) -> np.ndarray:
    """
    Assign each sample to one of three uncertainty strata based on entropy.

    0 = Low
    1 = Medium
    2 = High
    """
    strata = np.zeros(len(entropy), dtype=int)
    strata[entropy >= low_threshold] = 1
    strata[entropy >= high_threshold] = 2
    return strata


def stratum_labels(
    low_threshold: float = ENTROPY_LOW_THRESHOLD,
    high_threshold: float = ENTROPY_HIGH_THRESHOLD,
) -> List[Tuple[str, int]]:
    """Return display names for each stratum."""
    return [
        (f"Low (H<{low_threshold})", 0),
        (f"Medium ({low_threshold}≤H<{high_threshold})", 1),
        (f"High (H≥{high_threshold})", 2),
    ]


def uncertainty_report(
    probs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    low_threshold: float = ENTROPY_LOW_THRESHOLD,
    high_threshold: float = ENTROPY_HIGH_THRESHOLD,
    conf_threshold: float = CONF_UNCERTAIN_THRESHOLD,
) -> Dict:
    """
    Build an uncertainty-aware evaluation report for one set of predictions.
    """
    confidence = compute_confidence(probs)
    entropy = compute_entropy(probs)

    acc = accuracy_score(labels, predictions)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    uncertain_rate = float((confidence < conf_threshold).mean())
    mean_entropy = float(entropy.mean())

    strata = assign_strata(entropy, low_threshold, high_threshold)
    specs = stratum_labels(low_threshold, high_threshold)

    stratum_results = []
    for stratum_name, stratum_idx in specs:
        mask = strata == stratum_idx
        count = int(mask.sum())

        if count == 0:
            stratum_results.append({
                "stratum": stratum_name,
                "n": 0,
                "pct": 0.0,
                "accuracy": None,
                "mean_entropy": None,
                "mean_conf": None,
            })
            continue

        stratum_results.append({
            "stratum": stratum_name,
            "n": count,
            "pct": float(count / len(labels) * 100),
            "accuracy": float(accuracy_score(labels[mask], predictions[mask])),
            "mean_entropy": float(entropy[mask].mean()),
            "mean_conf": float(confidence[mask].mean()),
        })

    return {
        "n_samples": int(len(labels)),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "macro_f1": float(macro_f1),
        "uncertain_rate": uncertain_rate,
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": mean_entropy,
        "strata": stratum_results,
    }


def stratified_eval_fixed_baseline(
    baseline_probs: np.ndarray,
    baseline_labels: np.ndarray,
    baseline_preds: np.ndarray,
    augmented_preds: np.ndarray,
    low_threshold: float = ENTROPY_LOW_THRESHOLD,
    high_threshold: float = ENTROPY_HIGH_THRESHOLD,
) -> pd.DataFrame:
    """
    Evaluate baseline and comparison model on the same fixed strata,
    where the strata are defined exclusively by baseline entropy.
    """
    baseline_entropy = compute_entropy(baseline_probs)
    strata = assign_strata(
        baseline_entropy,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
    specs = stratum_labels(low_threshold, high_threshold)

    rows = []
    for stratum_name, stratum_idx in specs:
        mask = strata == stratum_idx
        count = int(mask.sum())

        if count == 0:
            rows.append({
                "Uncertainty Stratum": stratum_name,
                "Samples": 0,
                "% of Test": 0.0,
                "Acc. Baseline (%)": None,
                "Acc. Full Model (%)": None,
                "Δ Acc. (pp)": None,
            })
            continue

        acc_baseline = accuracy_score(baseline_labels[mask], baseline_preds[mask])
        acc_full = accuracy_score(baseline_labels[mask], augmented_preds[mask])

        rows.append({
            "Uncertainty Stratum": stratum_name,
            "Samples": count,
            "% of Test": round(count / len(baseline_labels) * 100, 1),
            "Acc. Baseline (%)": round(acc_baseline * 100, 1),
            "Acc. Full Model (%)": round(acc_full * 100, 1),
            "Δ Acc. (pp)": round((acc_full - acc_baseline) * 100, 1),
        })

    return pd.DataFrame(rows)


def stratified_eval(
    baseline_report: Dict,
    augmented_report: Dict,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper using precomputed stratum summaries.
    """
    rows = []
    for baseline_stratum, augmented_stratum in zip(
        baseline_report["strata"],
        augmented_report["strata"],
    ):
        acc_baseline = baseline_stratum["accuracy"]
        acc_augmented = augmented_stratum["accuracy"]

        if acc_baseline is None or acc_augmented is None:
            delta = None
        else:
            delta = (acc_augmented - acc_baseline) * 100

        rows.append({
            "stratum": baseline_stratum["stratum"],
            "n": baseline_stratum["n"],
            "pct (%)": round(baseline_stratum["pct"], 1),
            "acc_baseline": round(acc_baseline * 100, 1) if acc_baseline is not None else None,
            "acc_augmented": round(acc_augmented * 100, 1) if acc_augmented is not None else None,
            "Δ acc (pp)": round(delta, 1) if delta is not None else None,
        })

    return pd.DataFrame(rows)


def compare_configurations(configs: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a compact comparison DataFrame across configuration reports.
    """
    rows = []
    for name, report in configs.items():
        rows.append({
            "Configuration": name,
            "Accuracy (%)": round(report["accuracy"] * 100, 2),
            "Precision (%)": round(report["precision"] * 100, 2),
            "Recall (%)": round(report["recall"] * 100, 2),
            "F1-Score (%)": round(report["macro_f1"] * 100, 2),
            "Unc. Rate (%)": round(report["uncertain_rate"] * 100, 1),
            "Mean Entropy": round(report["mean_entropy"], 3),
        })
    return pd.DataFrame(rows)


def bootstrap_significance(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
    metric: str = "f1",
) -> Dict:
    """
    Paired bootstrap significance test for model A vs model B.
    """
    rng = np.random.default_rng(seed)
    sample_count = len(labels)

    def score(y_true, y_pred):
        if metric == "f1":
            _, _, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
            return f1
        return accuracy_score(y_true, y_pred)

    observed_diff = score(labels, preds_a) - score(labels, preds_b)

    diffs = []
    for _ in range(n_resamples):
        indices = rng.integers(0, sample_count, size=sample_count)
        diff = score(labels[indices], preds_a[indices]) - score(labels[indices], preds_b[indices])
        diffs.append(diff)

    diffs = np.array(diffs)
    p_value = float((diffs <= 0).mean())

    return {
        "observed_diff": float(observed_diff),
        "p_value": p_value,
        "significant": p_value < alpha,
        "alpha": alpha,
        "n_resamples": n_resamples,
    }


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error using max softmax confidence.
    """
    confidence = probs.max(axis=-1)
    total_samples = len(labels)
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidence >= low) & (confidence < high)
        if high == 1.0:
            mask = (confidence >= low) & (confidence <= high)

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        acc_bin = accuracy_score(labels[mask], predictions[mask])
        conf_bin = confidence[mask].mean()
        ece += (bin_count / total_samples) * abs(acc_bin - conf_bin)

    return float(ece)


def diagnose_uga_conditions(
    baseline_probs: np.ndarray,
    baseline_labels: np.ndarray,
    baseline_preds: np.ndarray,
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    alpha: float = ENTROPY_HIGH_THRESHOLD,
    beta: float = ENTROPY_LOW_THRESHOLD,
    n_classes: int = 3,
) -> Dict:
    """
    Diagnostic checks for uncertainty-guided augmentation.
    """
    report: Dict = {}

    train_entropy = compute_entropy(train_probs)
    mask_high = train_entropy >= alpha
    n_train = len(train_entropy)

    ece = compute_ece(baseline_probs, baseline_labels, baseline_preds)
    c3_pass = ece < 0.10
    report["C3_ece"] = round(ece * 100, 2)
    report["C3_pass"] = c3_pass
    report["C3_note"] = (
        f"ECE = {ece * 100:.2f}% | "
        f"{'PASS' if c3_pass else 'FAIL'}"
    )

    n_high = int(mask_high.sum())
    frac_high = n_high / n_train
    c4_pass = 0.05 <= frac_high <= 0.30
    report["C4_dhigh_n"] = n_high
    report["C4_dhigh_frac"] = round(frac_high * 100, 2)
    report["C4_pass"] = c4_pass
    report["C4_note"] = (
        f"D_high fraction = {frac_high * 100:.2f}% | "
        f"{'PASS' if c4_pass else 'FAIL'}"
    )

    if n_high > 0:
        high_labels = train_labels[mask_high]
        counts = np.bincount(high_labels, minlength=n_classes)
        nonzero = counts[counts > 0]
        imbalance = nonzero.max() / nonzero.min() if len(nonzero) > 1 else 1.0
        c5_pass = imbalance <= 3.0
        report["C5_class_counts"] = counts.tolist()
        report["C5_imbalance_ratio"] = round(float(imbalance), 2)
        report["C5_pass"] = c5_pass
        report["C5_note"] = (
            f"Imbalance ratio = {imbalance:.2f} | "
            f"{'PASS' if c5_pass else 'FAIL'}"
        )
    else:
        report["C5_pass"] = False
        report["C5_note"] = "D_high is empty."

    margin = compute_uncertainty_margin(train_probs)
    if n_high > 0:
        bimodal_frac = float((margin[mask_high] > 0.80).mean())
        report["C1_proxy_bimodal_frac"] = round(bimodal_frac * 100, 2)
        report["C1_note"] = (
            f"Bimodal fraction in D_high = {bimodal_frac * 100:.2f}%"
        )
    else:
        report["C1_note"] = "D_high is empty."

    report["C2_note"] = (
        "Manual inspection required: sample augmented sentences in D_high "
        "and verify sentiment polarity preservation."
    )

    hard_pass = report["C3_pass"] and report["C4_pass"] and report["C5_pass"]
    report["uga_recommended"] = hard_pass
    report["summary"] = (
        "UGA conditions passed."
        if hard_pass else
        "UGA conditions not fully satisfied."
    )

    return report


def print_uga_diagnostics(report: Dict) -> None:
    """
    Pretty-print UGA diagnostics.
    """
    print("\n" + "=" * 72)
    print("UGA PRE-DEPLOYMENT DIAGNOSTICS")
    print("=" * 72)
    for key in ["C1_note", "C2_note", "C3_note", "C4_note", "C5_note"]:
        if key in report:
            print(f"- {report[key]}")
    print(f"\nSummary: {report.get('summary', 'Unknown')}")
    print("=" * 72 + "\n")