"""
experiments/run_uncertainty_analysis.py
---------------------------------------
Generate uncertainty-stratified evaluation using baseline-fixed entropy strata.

This script must be run AFTER experiments/run_phobert.py has generated
results/*.json files.

Outputs:
  - table3_stratified_performance.csv
  - bootstrap_significance_uncertainty.json
  - uncertainty_arrays.csv
  - optional uncertainty_distribution.png
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.uncertainty import (
    stratified_eval_fixed_baseline,
    bootstrap_significance,
    compute_entropy,
    assign_strata,
    ENTROPY_LOW_THRESHOLD,
    ENTROPY_HIGH_THRESHOLD,
)

RESULTS_DIR = "results"

BASELINE_CONFIG = "PhoBERT - Baseline"
FULL_CONFIG = "PhoBERT + SR + BT + MC-Dropout"

CONFIG_ALIASES = {
    "PhoBERT + SR": "PhoBERT + Synonym Replacement",
    "PhoBERT + BT": "PhoBERT + Back-Translation",
    "PhoBERT + SR + BT": "PhoBERT + SR + BT (Full)",
    "PhoBERT + SR + BT + MC": "PhoBERT + SR + BT + MC-Dropout",
}


def normalize_config_name(name: str) -> str:
    return CONFIG_ALIASES.get(name, name)


def report_filename(name: str) -> str:
    return name.replace(" ", "_").replace("+", "_").replace("/", "_") + ".json"


def load_report(name: str, results_dir: str, seed_idx: int = 0) -> dict:
    canonical_name = normalize_config_name(name)
    path = os.path.join(results_dir, report_filename(canonical_name))

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Report not found: {path}\n"
            f"Run experiments/run_phobert.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        reports = json.load(f)

    if not isinstance(reports, list) or len(reports) == 0:
        raise ValueError(f"Invalid report format: {path}")

    if seed_idx >= len(reports):
        raise IndexError(
            f"seed_idx={seed_idx} out of range for {canonical_name}. "
            f"Available reports: {len(reports)}"
        )

    return reports[seed_idx]


def run_uncertainty_analysis(args):
    baseline_report = load_report(BASELINE_CONFIG, args.results_dir, args.seed_idx)
    full_report = load_report(FULL_CONFIG, args.results_dir, args.seed_idx)

    required_keys = ["mean_probs", "predictions", "labels"]
    for key in required_keys:
        if key not in baseline_report:
            raise ValueError(f"Baseline report missing key: '{key}'")
        if key not in full_report:
            raise ValueError(f"Full report missing key: '{key}'")

    baseline_probs = np.array(baseline_report["mean_probs"], dtype=np.float32)
    full_probs = np.array(full_report["mean_probs"], dtype=np.float32)

    baseline_preds = np.array(baseline_report["predictions"], dtype=np.int64)
    full_preds = np.array(full_report["predictions"], dtype=np.int64)

    baseline_labels = np.array(baseline_report["labels"], dtype=np.int64)
    full_labels = np.array(full_report["labels"], dtype=np.int64)

    if len(baseline_labels) != len(full_labels):
        raise ValueError("Baseline and full reports have different label lengths.")

    if not np.array_equal(baseline_labels, full_labels):
        raise ValueError(
            "Baseline and full reports do not share the same label ordering. "
            "Stratified comparison requires identical test ordering."
        )

    if len(baseline_labels) != len(baseline_probs) or len(baseline_labels) != len(full_probs):
        raise ValueError("Mismatch between labels and probability arrays.")

    # Correct call: baseline_probs, labels, baseline_preds, full_preds
    table3 = stratified_eval_fixed_baseline(
        baseline_probs=baseline_probs,
        baseline_labels=baseline_labels,
        baseline_preds=baseline_preds,
        augmented_preds=full_preds,
        low_threshold=ENTROPY_LOW_THRESHOLD,
        high_threshold=ENTROPY_HIGH_THRESHOLD,
    )

    print("\n" + "=" * 72)
    print("TABLE 3 — UNCERTAINTY-STRATIFIED PERFORMANCE (BASELINE-FIXED STRATA)")
    print("=" * 72)
    print(table3.to_string(index=False))

    table3_path = os.path.join(args.results_dir, "table3_stratified_performance.csv")
    table3.to_csv(table3_path, index=False)
    print(f"\nSaved -> {table3_path}")

    baseline_entropy = compute_entropy(baseline_probs)
    full_entropy = compute_entropy(full_probs)
    strata = assign_strata(baseline_entropy)

    print("\n" + "=" * 72)
    print("BASELINE ENTROPY DISTRIBUTION")
    print("=" * 72)
    print(
        f"Thresholds: Low < {ENTROPY_LOW_THRESHOLD} | "
        f"Medium [{ENTROPY_LOW_THRESHOLD}, {ENTROPY_HIGH_THRESHOLD}) | "
        f"High >= {ENTROPY_HIGH_THRESHOLD}"
    )

    stratum_names = ["Low", "Medium", "High"]
    for idx, name in enumerate(stratum_names):
        count = int((strata == idx).sum())
        pct = count / len(strata) * 100
        print(f"{name:>6}: {count:4d} samples ({pct:5.1f}%)")

    print("\n" + "=" * 72)
    print("BOOTSTRAP SIGNIFICANCE TESTS")
    print("=" * 72)

    sig_main = bootstrap_significance(
        labels=baseline_labels,
        preds_a=full_preds,
        preds_b=baseline_preds,
    )

    print("Full vs Baseline")
    print(f"  Δ Macro-F1 = {sig_main['observed_diff'] * 100:+.2f} pp")
    print(f"  p-value    = {sig_main['p_value']:.4f}")
    print(f"  Significant at α=0.05: {sig_main['significant']}")

    significance_outputs = {
        "full_vs_baseline": sig_main
    }

    for config_name in ["PhoBERT + Synonym Replacement", "PhoBERT + Back-Translation"]:
        try:
            rep = load_report(config_name, args.results_dir, args.seed_idx)
            preds = np.array(rep["predictions"], dtype=np.int64)

            extra_sig = bootstrap_significance(
                labels=baseline_labels,
                preds_a=full_preds,
                preds_b=preds,
            )

            print(f"\nFull vs {config_name}")
            print(f"  Δ Macro-F1 = {extra_sig['observed_diff'] * 100:+.2f} pp")
            print(f"  p-value    = {extra_sig['p_value']:.4f}")
            print(f"  Significant at α=0.05: {extra_sig['significant']}")

            significance_outputs[f"full_vs_{config_name}"] = extra_sig

        except Exception as exc:
            print(f"\n[SKIP] {config_name}: {exc}")

    sig_path = os.path.join(args.results_dir, "bootstrap_significance_uncertainty.json")
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(significance_outputs, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {sig_path}")

    unc_df = pd.DataFrame({
        "true_label": baseline_labels,
        "pred_baseline": baseline_preds,
        "pred_full": full_preds,
        "entropy_baseline": baseline_entropy,
        "entropy_full": full_entropy,
        "stratum_baseline": strata,
        "correct_baseline": (baseline_preds == baseline_labels).astype(int),
        "correct_full": (full_preds == baseline_labels).astype(int),
    })

    arrays_path = os.path.join(args.results_dir, "uncertainty_arrays.csv")
    unc_df.to_csv(arrays_path, index=False)
    print(f"Saved -> {arrays_path}")

    if args.plot:
        plot_uncertainty_histograms(
            baseline_entropy=baseline_entropy,
            full_entropy=full_entropy,
            results_dir=args.results_dir,
        )


def plot_uncertainty_histograms(baseline_entropy, full_entropy, results_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available -> skipping")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, values, title in zip(
        axes,
        [baseline_entropy, full_entropy],
        ["Baseline entropy", "Full model entropy"],
    ):
        ax.hist(values, bins=30, edgecolor="white", alpha=0.9)
        ax.axvline(
            ENTROPY_LOW_THRESHOLD,
            linestyle="--",
            linewidth=1.2,
            label="Low threshold"
        )
        ax.axvline(
            ENTROPY_HIGH_THRESHOLD,
            linestyle="--",
            linewidth=1.2,
            label="High threshold"
        )
        ax.set_title(title)
        ax.set_xlabel("Predictive entropy")
        ax.set_ylabel("Count")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(results_dir, "uncertainty_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Entropy-based uncertainty-stratified evaluation"
    )
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument(
        "--seed_idx",
        type=int,
        default=0,
        help="Index of the run to inspect in the saved JSON list."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate uncertainty histograms"
    )
    args = parser.parse_args()
    run_uncertainty_analysis(args)


if __name__ == "__main__":
    main()