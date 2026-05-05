"""
experiments/run_phobert.py
--------------------------
Run PhoBERT / mBERT sentiment experiments with 5-fold cross-validation.

This script produces:
  - Per-configuration JSON reports
  - table2_overall_performance.csv
  - table3_stratified_performance.csv
  - table4_calibration.csv
  - bootstrap_significance.json

Design choices:
  - Default seeds are set to 5 runs to match the manuscript.
  - Stratified analysis compares:
        PhoBERT - Baseline
    vs  PhoBERT + SR + BT + MC-Dropout
  - Entropy strata are fixed by the baseline model only.
"""

import os
import sys
import json
import argparse
import shutil
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import (
    create_5fold_splits,
    load_augmented_for_fold,
    VLSPDataset,
    verify_no_data_leakage,
)
from src.models.phobert_classifier import PhoBERTTrainer, PHOBERT_BASE, MBERT_BASE
from src.evaluation.uncertainty import (
    uncertainty_report,
    stratified_eval_fixed_baseline,
    bootstrap_significance,
    compute_ece,
    ENTROPY_LOW_THRESHOLD,
    ENTROPY_HIGH_THRESHOLD,
)

RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CONFIGS = {
    "PhoBERT - Baseline": (PHOBERT_BASE, "baseline"),
    "PhoBERT + Synonym Replacement": (PHOBERT_BASE, "sr"),
    "PhoBERT + Back-Translation": (PHOBERT_BASE, "bt"),
    "mBERT - Baseline": (MBERT_BASE, "baseline"),
    "mBERT + SR + BT": (MBERT_BASE, "sr+bt"),
    "PhoBERT + SR + BT (Full)": (PHOBERT_BASE, "sr+bt"),
    "PhoBERT + SR + BT + MC-Dropout": (PHOBERT_BASE, "sr+bt"),
}

CONFIG_ALIASES = {
    "PhoBERT + SR": "PhoBERT + Synonym Replacement",
    "PhoBERT + BT": "PhoBERT + Back-Translation",
    "PhoBERT + SR + BT": "PhoBERT + SR + BT (Full)",
    "PhoBERT + SR + BT + MC": "PhoBERT + SR + BT + MC-Dropout",
}

MC_CONFIG = "PhoBERT + SR + BT + MC-Dropout"
BASELINE_CONFIG = "PhoBERT - Baseline"
FULL_CONFIG = "PhoBERT + SR + BT + MC-Dropout"


def normalize_config_name(name: str) -> str:
    return CONFIG_ALIASES.get(name, name)


def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("+", "_").replace("/", "_")


def _pool_cv_mean_probs(reports: List[Dict], n_folds: int, n_seeds: int):
    """Average mean probabilities across seeds within each fold, then pool folds."""
    pooled_probs, pooled_labels = [], []
    for fold_idx in range(n_folds):
        start = fold_idx * n_seeds
        end = start + n_seeds
        fold_reports = reports[start:end]
        if not fold_reports:
            continue
        fold_probs = np.mean([np.array(r["mean_probs"]) for r in fold_reports], axis=0)
        fold_labels = np.array(fold_reports[0]["labels"])
        pooled_probs.append(fold_probs)
        pooled_labels.append(fold_labels)
    if not pooled_probs:
        return np.empty((0, 3)), np.empty((0,), dtype=int)
    return np.vstack(pooled_probs), np.concatenate(pooled_labels)


def aggregate_fold_seed_reports(reports: List[Dict]) -> Dict:
    metrics = [
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "uncertain_rate",
        "mean_entropy",
        "ece",
    ]
    agg = {}
    for metric in metrics:
        values = [r[metric] for r in reports if metric in r]
        if values:
            agg[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return agg


def fmt_mean_std(agg: Dict, metric: str, pct: bool = True, decimals: int = 2) -> str:
    if metric not in agg:
        return "—"
    scale = 100 if pct else 1
    mean = agg[metric]["mean"] * scale
    std = agg[metric]["std"] * scale
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def run_single_fold_seed(
    config_name: str,
    model_name: str,
    aug_config: str,
    fold: Dict,
    fold_idx: int,
    seed: int,
    data_dir: str,
    output_dir: str,
    mc_passes: int = 20,
) -> Dict:
    import torch
    from sklearn.utils.class_weight import compute_class_weight
    from src.utils.text_preprocessing import segment_text

    print(f"\n{'=' * 60}")
    print(f"Config : {config_name}")
    print(f"Fold   : {fold_idx + 1}/5")
    print(f"Seed   : {seed}")
    print(f"{'=' * 60}")

    train_df = load_augmented_for_fold(data_dir, fold["train"], aug_config, fold_idx=fold_idx)
    val_df = fold["val"]
    test_df = fold["test"]

    verify_no_data_leakage(fold["train"], test_df)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"].values),
        y=train_df["label"].values,
    )
    print(f"[Class Weights] {dict(enumerate(class_weights.round(3)))}")

    checkpoint_name = f"{safe_name(config_name)}_fold{fold_idx}_seed{seed}"
    checkpoint_path = os.path.join(output_dir, checkpoint_name)

    trainer = PhoBERTTrainer(
        model_name=model_name,
        lr=2e-5,
        epochs=10,
        batch_size=32,
        max_length=128,
        patience=3,
        seed=seed,
        output_dir=checkpoint_path,
        class_weights=class_weights.tolist(),
    )

    train_texts = [segment_text(text) for text in train_df["text"].tolist()]
    val_texts = [segment_text(text) for text in val_df["text"].tolist()]
    test_texts = [segment_text(text) for text in test_df["text"].tolist()]

    train_ds = VLSPDataset(train_texts, train_df["label"], trainer.tokenizer, segment=False)
    val_ds = VLSPDataset(val_texts, val_df["label"], trainer.tokenizer, segment=False)
    test_ds = VLSPDataset(test_texts, test_df["label"], trainer.tokenizer, segment=False)

    train_result = trainer.train(train_ds, val_ds)
    print(f"Best val F1: {train_result['best_val_f1']:.4f}")

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    if config_name == MC_CONFIG:
        result = trainer.predict_with_uncertainty(test_loader, mc_passes=mc_passes)
    else:
        trainer.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(trainer.device) for k, v in batch.items()}
                labels = batch.pop("labels")
                logits = trainer.model(**batch)["logits"]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())

        probs_arr = np.array(all_probs)
        labels_arr = np.array(all_labels)

        result = {
            "predictions": probs_arr.argmax(axis=-1),
            "mean_probs": probs_arr,
            "labels": labels_arr,
        }

    report = uncertainty_report(
        probs=np.array(result["mean_probs"]),
        labels=np.array(result["labels"]),
        predictions=np.array(result["predictions"]),
    )
    report["ece"] = compute_ece(
        probs=np.array(result["mean_probs"]),
        labels=np.array(result["labels"]),
        predictions=np.array(result["predictions"]),
    )
    report["config"] = config_name
    report["model"] = model_name
    report["aug_config"] = aug_config
    report["seed"] = seed
    report["fold"] = fold_idx
    report["predictions"] = result["predictions"].tolist()
    report["labels"] = result["labels"].tolist()
    report["mean_probs"] = result["mean_probs"].tolist()
    report["best_val_f1"] = float(train_result["best_val_f1"])

    return report


def save_config_reports(config_name: str, reports: List[Dict]):
    out_path = os.path.join(RESULTS_DIR, f"{safe_name(config_name)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {out_path}")


def print_summary(all_reports: Dict[str, List[Dict]]):
    print("\n" + "=" * 72)
    print("AGGREGATED RESULTS (mean ± std across folds × seeds)")
    print("=" * 72)
    for config_name, reports in all_reports.items():
        agg = aggregate_fold_seed_reports(reports)
        print(f"\n{config_name}")
        for metric, stats in agg.items():
            scale = 100 if metric != "mean_entropy" else 1
            print(f"  {metric:20s}: {stats['mean'] * scale:.2f} ± {stats['std'] * scale:.2f}")


def build_table2_overall_performance(all_reports: Dict[str, List[Dict]]):
    rows = []
    for config_name, reports in all_reports.items():
        agg = aggregate_fold_seed_reports(reports)

        rows.append({
            "Configuration": config_name,
            "Acc. (%)": fmt_mean_std(agg, "accuracy"),
            "Prec. (%)": fmt_mean_std(agg, "precision"),
            "Rec. (%)": fmt_mean_std(agg, "recall"),
            "F1 (%)": fmt_mean_std(agg, "macro_f1"),
            "Unc. Rate (%)": fmt_mean_std(agg, "uncertain_rate", decimals=2),
            "Mean Ent.": fmt_mean_std(agg, "mean_entropy", pct=False, decimals=4),
            "ECE (%)": fmt_mean_std(agg, "ece"),
        })

    table2 = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "table2_overall_performance.csv")
    table2.to_csv(path, index=False)
    print("\n=== TABLE 2: OVERALL PERFORMANCE ===")
    print(table2.to_string(index=False))
    print(f"[Saved] {path}")


def build_table3_stratified(all_reports: Dict[str, List[Dict]], args):
    if BASELINE_CONFIG not in all_reports or FULL_CONFIG not in all_reports:
        print("[Skip] Table 3 cannot be built because required configs are missing.")
        return

    baseline_probs, baseline_labels = _pool_cv_mean_probs(
        all_reports[BASELINE_CONFIG], args.n_folds, len(args.seeds)
    )
    full_probs, _ = _pool_cv_mean_probs(
        all_reports[FULL_CONFIG], args.n_folds, len(args.seeds)
    )

    baseline_preds = baseline_probs.argmax(axis=-1)
    full_preds = full_probs.argmax(axis=-1)

    table3 = stratified_eval_fixed_baseline(
        baseline_probs=baseline_probs,
        baseline_labels=baseline_labels,
        baseline_preds=baseline_preds,
        augmented_preds=full_preds,
        low_threshold=ENTROPY_LOW_THRESHOLD,
        high_threshold=ENTROPY_HIGH_THRESHOLD,
    )

    path = os.path.join(RESULTS_DIR, "table3_stratified_performance.csv")
    table3.to_csv(path, index=False)

    print("\n=== TABLE 3: UNCERTAINTY-STRATIFIED PERFORMANCE ===")
    print(
        f"[Thresholds] Low < {ENTROPY_LOW_THRESHOLD} | "
        f"Medium [{ENTROPY_LOW_THRESHOLD}, {ENTROPY_HIGH_THRESHOLD}) | "
        f"High >= {ENTROPY_HIGH_THRESHOLD}"
    )
    print(table3.to_string(index=False))
    print(f"[Saved] {path}")


def build_table4_calibration(all_reports: Dict[str, List[Dict]]):
    rows = []

    for config_name in [BASELINE_CONFIG, FULL_CONFIG]:
        if config_name not in all_reports:
            continue

        agg = aggregate_fold_seed_reports(all_reports[config_name])
        rows.append({
            "Model": config_name,
            "ECE ↓ (%)": fmt_mean_std(agg, "ece"),
        })

    table4 = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "table4_calibration.csv")
    table4.to_csv(path, index=False)

    print("\n=== TABLE 4: CALIBRATION ===")
    print(table4.to_string(index=False))
    print(f"[Saved] {path}")


def run_bootstrap_tests(all_reports: Dict[str, List[Dict]], args):
    if BASELINE_CONFIG not in all_reports or FULL_CONFIG not in all_reports:
        print("[Skip] Bootstrap tests cannot be built because required configs are missing.")
        return

    baseline_probs, labels = _pool_cv_mean_probs(
        all_reports[BASELINE_CONFIG], args.n_folds, len(args.seeds)
    )
    full_probs, _ = _pool_cv_mean_probs(
        all_reports[FULL_CONFIG], args.n_folds, len(args.seeds)
    )

    baseline_preds = baseline_probs.argmax(axis=-1)
    full_preds = full_probs.argmax(axis=-1)

    results = {
        "full_vs_baseline": bootstrap_significance(
            labels=np.array(labels),
            preds_a=np.array(full_preds),
            preds_b=np.array(baseline_preds),
        )
    }

    for config_name in ["PhoBERT + Synonym Replacement", "PhoBERT + Back-Translation"]:
        if config_name not in all_reports:
            continue

        other_probs, _ = _pool_cv_mean_probs(
            all_reports[config_name], args.n_folds, len(args.seeds)
        )
        other_preds = other_probs.argmax(axis=-1)
        results[f"full_vs_{config_name}"] = bootstrap_significance(
            labels=np.array(labels),
            preds_a=np.array(full_preds),
            preds_b=np.array(other_preds),
        )

    path = os.path.join(RESULTS_DIR, "bootstrap_significance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== BOOTSTRAP SIGNIFICANCE ===")
    for name, stat in results.items():
        print(
            f"{name}: Δ={stat['observed_diff'] * 100:+.2f} pp | "
            f"p={stat['p_value']:.4f} | significant={stat['significant']}"
        )
    print(f"[Saved] {path}")


def run_all(args):
    config_name = normalize_config_name(args.config)

    folds = create_5fold_splits(
        data_dir=args.data_dir,
        n_splits=args.n_folds,
        random_state=42,
    )

    if config_name == "all":
        configs_to_run = list(CONFIGS.items())
    else:
        if config_name not in CONFIGS:
            raise ValueError(
                f"Unknown config: {args.config}\n"
                f"Accepted configs: {list(CONFIGS.keys())}"
            )
        configs_to_run = [(config_name, CONFIGS[config_name])]

    all_reports = {}
    best_global_checkpoints = {}

    for config_name, (model_name, aug_config) in configs_to_run:
        print(f"\n{'#' * 72}")
        print(f"CONFIG: {config_name}")
        print(f"{'#' * 72}")

        config_reports = []

        for fold_data in folds:
            fold_idx = fold_data["fold"]

            for seed in args.seeds:
                report = run_single_fold_seed(
                    config_name=config_name,
                    model_name=model_name,
                    aug_config=aug_config,
                    fold=fold_data,
                    fold_idx=fold_idx,
                    seed=seed,
                    data_dir=args.data_dir,
                    output_dir=CHECKPOINT_DIR,
                    mc_passes=args.mc_passes,
                )

                checkpoint_name = f"{safe_name(config_name)}_fold{fold_idx}_seed{seed}"
                checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
                current_score = report["best_val_f1"]

                if config_name not in best_global_checkpoints:
                    best_global_checkpoints[config_name] = {
                        "score": current_score,
                        "path": checkpoint_path,
                    }
                    print(
                        f"[Checkpoint Manager] Initialize best for {config_name}: "
                        f"{current_score:.4f}"
                    )
                else:
                    best_info = best_global_checkpoints[config_name]
                    if current_score > best_info["score"]:
                        print(
                            f"[Checkpoint Manager] New best for {config_name}: "
                            f"{current_score:.4f} > {best_info['score']:.4f}"
                        )
                        if os.path.exists(best_info["path"]):
                            shutil.rmtree(best_info["path"])
                        best_global_checkpoints[config_name] = {
                            "score": current_score,
                            "path": checkpoint_path,
                        }
                    else:
                        print(
                            f"[Checkpoint Manager] Delete lower-scoring checkpoint: "
                            f"{current_score:.4f} <= {best_info['score']:.4f}"
                        )
                        if os.path.exists(checkpoint_path):
                            shutil.rmtree(checkpoint_path)

                config_reports.append(report)

        all_reports[config_name] = config_reports
        save_config_reports(config_name, config_reports)

    with open(os.path.join(RESULTS_DIR, "all_reports.json"), "w", encoding="utf-8") as f:
        json.dump(all_reports, f, ensure_ascii=False, indent=2)

    print_summary(all_reports)
    build_table2_overall_performance(all_reports)
    build_table3_stratified(all_reports, args)
    build_table4_calibration(all_reports)
    run_bootstrap_tests(all_reports, args)

    print(f"\nAll outputs saved under: {RESULTS_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Run PhoBERT sentiment experiments (5-fold CV, entropy stratification)"
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Root data directory containing raw/ and augmented/"
    )
    parser.add_argument(
        "--config",
        default="all",
        help="Configuration name or 'all'"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2024, 3407, 5555],
        help="Random seeds. Default uses 5 runs to match the manuscript."
    )
    parser.add_argument(
        "--mc_passes",
        type=int,
        default=20,
        help="Number of MC-Dropout passes"
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds (default: 5)"
    )
    args = parser.parse_args()

    print(f"\nExperiment start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Folds: {args.n_folds} | Seeds: {args.seeds} | MC passes: {args.mc_passes}")
    run_all(args)
    print(f"\nExperiment end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()