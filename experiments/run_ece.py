"""
experiments/run_ece.py
----------------------
Expected Calibration Error (ECE) + Temperature Scaling evaluation.

Nguyên tắc đúng:
- Temperature scaling chỉ được học trên validation set độc lập.
- Nếu report không chứa val_probs/val_labels thì KHÔNG dùng test set để học T*.
- Trong trường hợp thiếu validation probabilities, chỉ báo cáo ECE/MCE raw.

Usage:
    python experiments/run_ece.py --results_dir results

Outputs:
    results/table8_calibration.csv
    results/calibration_report.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── ECE / MCE computation ─────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error.

    ECE = (1/N) * Σ_m |B_m| * |acc(B_m) − conf(B_m)|
    """
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Maximum Calibration Error.
    """
    confidences = probs.max(axis=-1)
    predictions = probs.argmax(axis=-1)
    correct = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        mce = max(mce, abs(bin_acc - bin_conf))

    return float(mce)


# ── Temperature Scaling ───────────────────────────────────────────────────────

def find_optimal_temperature(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    t_range: tuple = (0.1, 5.0),
    n_steps: int = 100,
) -> float:
    """
    Find T* that minimises NLL on validation set via grid search.

    Vì chỉ có probabilities (không có logits), ta dùng xấp xỉ:
    probs^(1/T) rồi renormalise.
    """
    best_t = 1.0
    best_nll = float("inf")
    eps = 1e-10

    for t in np.linspace(t_range[0], t_range[1], n_steps):
        scaled = np.power(val_probs + eps, 1.0 / t)
        scaled = scaled / scaled.sum(axis=-1, keepdims=True)
        nll = -np.log(scaled[np.arange(len(val_labels)), val_labels] + eps).mean()
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    return best_t


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to probabilities.
    """
    eps = 1e-10
    scaled = np.power(probs + eps, 1.0 / temperature)
    return scaled / scaled.sum(axis=-1, keepdims=True)


# ── Load saved results ────────────────────────────────────────────────────────

def _report_filename(config_name: str) -> str:
    return config_name.replace(" ", "_").replace("+", "_").replace("/", "_") + ".json"


def load_report(config_name: str, results_dir: str, seed_idx: int = 0) -> dict:
    path = os.path.join(results_dir, _report_filename(config_name))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Not found: {path}\n"
            f"Run run_phobert.py first and ensure mean_probs is saved."
        )
    with open(path, "r", encoding="utf-8") as f:
        reports = json.load(f)

    if not isinstance(reports, list) or len(reports) == 0:
        raise ValueError(f"Invalid report format in {path}")

    if seed_idx >= len(reports):
        raise IndexError(f"seed_idx={seed_idx} out of range for {path}")

    return reports[seed_idx]


# ── Main evaluation ───────────────────────────────────────────────────────────

def calibration_status(ece_value: float) -> str:
    if ece_value < 0.04:
        return "Well-calibrated"
    if ece_value < 0.07:
        return "Good"
    return "Needs improvement"


def run_ece_evaluation(args):
    configs = [
        ("PhoBERT - Baseline", "PhoBERT baseline"),
        ("PhoBERT + SR + BT (Full)", "SR+BT (no scaling)"),
        ("PhoBERT + SR + BT + MC-Dropout", "SR+BT + MC (no scaling)"),
    ]

    rows = []
    details = {}

    for config_key, config_label in configs:
        try:
            report = load_report(config_key, args.results_dir, seed_idx=args.seed_idx)
        except Exception as exc:
            print(f"[SKIP] {config_key}: {exc}")
            continue

        if "mean_probs" not in report or "labels" not in report:
            print(f"[SKIP] {config_key}: report thiếu mean_probs hoặc labels")
            continue

        probs = np.array(report["mean_probs"], dtype=np.float32)
        labels = np.array(report["labels"], dtype=np.int64)

        ece_raw = compute_ece(probs, labels, n_bins=args.n_bins)
        mce_raw = compute_mce(probs, labels, n_bins=args.n_bins)
        f1_raw = float(report.get("macro_f1", np.nan))

        print(f"\n{'=' * 60}")
        print(config_label)
        print(f"  ECE (before scaling): {ece_raw * 100:.2f}%")
        print(f"  MCE (before scaling): {mce_raw * 100:.2f}%")
        print(f"  F1:                   {f1_raw * 100:.2f}%")

        # Chỉ scale nếu có validation probs thật sự
        has_val = "val_probs" in report and "val_labels" in report
        if has_val:
            val_probs = np.array(report["val_probs"], dtype=np.float32)
            val_labels = np.array(report["val_labels"], dtype=np.int64)

            temperature = find_optimal_temperature(val_probs, val_labels)
            probs_scaled = apply_temperature(probs, temperature)
            ece_scaled = compute_ece(probs_scaled, labels, n_bins=args.n_bins)
            mce_scaled = compute_mce(probs_scaled, labels, n_bins=args.n_bins)

            print(f"  T*: {temperature:.3f}")
            print(f"  ECE (after  scaling): {ece_scaled * 100:.2f}%")
            print(f"  MCE (after  scaling): {mce_scaled * 100:.2f}%")
            print(f"  F1  (after  scaling): {f1_raw * 100:.2f}% (unchanged)")
        else:
            temperature = None
            ece_scaled = None
            mce_scaled = None
            print("  [Note] val_probs/val_labels not found -> skip temperature scaling")

        row = {
            "Model / Configuration": config_label,
            "ECE (%)": round(ece_raw * 100, 2),
            "MCE (%)": round(mce_raw * 100, 2),
            "F1 (%)": round(f1_raw * 100, 2),
            "Calibration Status": calibration_status(ece_raw),
        }

        if temperature is not None:
            row["T*"] = round(temperature, 3)
            row["ECE scaled (%)"] = round(ece_scaled * 100, 2)
            row["MCE scaled (%)"] = round(mce_scaled * 100, 2)
            row["Scaled Status"] = calibration_status(ece_scaled)
        else:
            row["T*"] = "N/A"
            row["ECE scaled (%)"] = "N/A"
            row["MCE scaled (%)"] = "N/A"
            row["Scaled Status"] = "N/A"

        rows.append(row)

        details[config_label] = {
            "ece_raw": ece_raw,
            "mce_raw": mce_raw,
            "ece_scaled": ece_scaled,
            "mce_scaled": mce_scaled,
            "temperature": temperature,
            "f1": f1_raw,
            "used_validation_scaling": has_val,
        }

    if rows:
        table8 = pd.DataFrame(rows)
        out_csv = os.path.join(args.results_dir, "table8_calibration.csv")
        table8.to_csv(out_csv, index=False)

        print("\n" + "=" * 60)
        print("TABLE 8 — Calibration Results")
        print("=" * 60)
        print(table8.to_string(index=False))
        print(f"\nSaved -> {out_csv}")

        out_json = os.path.join(args.results_dir, "calibration_report.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"Details -> {out_json}")
    else:
        print("\n[WARNING] No valid reports loaded. Run run_phobert.py first.")

    return details


def main():
    parser = argparse.ArgumentParser(description="ECE + Temperature Scaling Evaluation")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--n_bins", type=int, default=15,
                        help="Number of calibration bins")
    parser.add_argument("--seed_idx", type=int, default=0,
                        help="Index of report in results JSON list")
    args = parser.parse_args()
    run_ece_evaluation(args)


if __name__ == "__main__":
    main()