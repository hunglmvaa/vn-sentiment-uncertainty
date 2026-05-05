"""
experiments/run_uit_vsfc.py
-----------------------------
Zero-shot cross-domain evaluation on UIT-VSFC.

Loads the best VLSP 2016 model (PhoBERT + SR + BT + MC-Dropout)
and evaluates it on UIT-VSFC WITHOUT any fine-tuning.

This is the cross-domain transfer experiment described in Section 4.6.

Dataset: uitnlp/vietnamese_students_feedback (HuggingFace)
  - 16,175 sentences, 3-class sentiment (identical schema to VLSP 2016)
  - Positive=2, Negative=0, Neutral=1

Usage:
    python experiments/run_uit_vsfc.py \
        --checkpoint results/checkpoints/PhoBERT_–_SR_BT_MC-Dropout_42/best_model \
        --mc_passes 20

Outputs:
    results/uit_vsfc_evaluation.csv
    results/uit_vsfc_stratified.csv
    results/uit_vsfc_report.json
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.phobert_classifier import PhoBERTTrainer, PHOBERT_BASE
from src.utils.data_loader import VLSPDataset
from src.evaluation.uncertainty import (
    uncertainty_report, stratified_eval, bootstrap_significance
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# UIT-VSFC label mapping (same schema as VLSP 2016)
VIHSD_LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}


def load_uit_vsfc(split: str = "test", cache_dir: str = None) -> pd.DataFrame:
    """
    Load UIT-VSFC from HuggingFace datasets library.

    uitnlp/vietnamese_students_feedback
    Columns: sentence, sentiment (0=negative, 1=neutral, 2=positive)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install HuggingFace datasets: pip install datasets"
        )

    print(f"[UIT-VSFC] Loading '{split}' split from HuggingFace...")
    ds = load_dataset(
        "uitnlp/vietnamese_students_feedback",
        split=split,
        cache_dir=cache_dir,
    )

    df = pd.DataFrame({
        "text":  ds["sentence"],
        "label": ds["sentiment"],   # already 0/1/2
    })
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    pos = (df.label == 2).sum()
    neu = (df.label == 1).sum()
    neg = (df.label == 0).sum()
    print(f"[UIT-VSFC] {split}: {len(df)} samples "
          f"(pos={pos}, neu={neu}, neg={neg})")
    return df


def evaluate_on_vsfc(
    trainer:    PhoBERTTrainer,
    vsfc_df:    pd.DataFrame,
    mc_passes:  int = 20,
    batch_size: int = 32,
) -> Dict:
    """
    Run MC-Dropout inference on UIT-VSFC test set and return full report.
    No fine-tuning — pure zero-shot cross-domain transfer.
    """
    from torch.utils.data import DataLoader

    test_ds = VLSPDataset(vsfc_df["text"], vsfc_df["label"], trainer.tokenizer)
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    result  = trainer.predict_with_uncertainty(loader, mc_passes=mc_passes)

    report  = uncertainty_report(
        probs       = result["mean_probs"],
        labels      = result["labels"],
        predictions = result["predictions"],
    )
    report["mean_probs"]  = result["mean_probs"].tolist()
    report["predictions"] = result["predictions"].tolist()
    report["labels"]      = result["labels"].tolist()
    return report


def run_uit_vsfc_evaluation(args):
    print("\n" + "="*60)
    print("UIT-VSFC Zero-Shot Cross-Domain Evaluation")
    print("="*60)

    # ── Load VLSP 2016 trained model ──────────────────────────────────────────
    trainer = PhoBERTTrainer(
        model_name = PHOBERT_BASE,
        output_dir = os.path.dirname(args.checkpoint),
    )

    ckpt_model_pt = os.path.join(args.checkpoint, "model.pt")
    if not os.path.exists(ckpt_model_pt):
        print(f"[ERROR] Checkpoint not found: {ckpt_model_pt}")
        print("Run run_phobert.py first with config 'PhoBERT + SR + BT + MC-Dropout'")
        return

    trainer._load_checkpoint(os.path.basename(args.checkpoint))
    print(f"[Model] Loaded from {args.checkpoint}")

    # ── Load UIT-VSFC ─────────────────────────────────────────────────────────
    try:
        vsfc_test = load_uit_vsfc("test", cache_dir=args.cache_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load UIT-VSFC: {e}")
        return

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\n[Evaluation] Running MC-Dropout on UIT-VSFC (zero-shot)...")
    vsfc_report = evaluate_on_vsfc(trainer, vsfc_test, mc_passes=args.mc_passes)

    print("\n" + "─"*50)
    print(f"  UIT-VSFC Test Results (zero-shot cross-domain):")
    print(f"  Accuracy:      {vsfc_report['accuracy']*100:.2f}%")
    print(f"  Macro F1:      {vsfc_report['macro_f1']*100:.2f}%")
    print(f"  Precision:     {vsfc_report['precision']*100:.2f}%")
    print(f"  Recall:        {vsfc_report['recall']*100:.2f}%")
    print(f"  Unc. Rate:     {vsfc_report['uncertain_rate']*100:.1f}%")
    print(f"  Mean Entropy:  {vsfc_report['mean_entropy']:.4f}")

    # ── Per-stratum breakdown ─────────────────────────────────────────────────
    print("\n  Uncertainty-Stratified Breakdown:")
    for s in vsfc_report["strata"]:
        if s["n"] > 0:
            print(f"    {s['stratum']}: "
                  f"n={s['n']} ({s['pct']:.1f}%), "
                  f"acc={s['accuracy']*100:.1f}%")

    # ── Compare with VLSP 2016 baseline ──────────────────────────────────────
    vlsp_baseline_path = os.path.join(
        args.results_dir,
        "PhoBERT_–_Baseline.json"
    )
    if os.path.exists(vlsp_baseline_path):
        with open(vlsp_baseline_path) as f:
            vlsp_baseline = json.load(f)[0]

        print("\n  Comparison vs VLSP 2016 baseline:")
        print(f"    VLSP 2016 (in-domain):  F1={vlsp_baseline['macro_f1']*100:.2f}%")
        print(f"    UIT-VSFC (zero-shot):   F1={vsfc_report['macro_f1']*100:.2f}%")
        delta = (vsfc_report['macro_f1'] - vlsp_baseline['macro_f1']) * 100
        print(f"    Domain shift gap:       {delta:+.2f} pp")

    # ── Also load VLSP full model for comparison ──────────────────────────────
    vlsp_full_path = os.path.join(
        args.results_dir,
        "PhoBERT___SR___BT_(Full).json"
    )
    if os.path.exists(vlsp_full_path):
        with open(vlsp_full_path) as f:
            vlsp_full = json.load(f)[0]
        print(f"    VLSP 2016 SR+BT (in-domain): F1={vlsp_full['macro_f1']*100:.2f}%")

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = os.path.join(args.results_dir, "uit_vsfc_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(vsfc_report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report → {out_json}")

    # Table: cross-domain summary
    table_row = pd.DataFrame([{
        "Dataset":          "UIT-VSFC",
        "Setting":          "Zero-shot cross-domain",
        "Accuracy (%)":     round(vsfc_report["accuracy"] * 100, 2),
        "Precision (%)":    round(vsfc_report["precision"] * 100, 2),
        "Recall (%)":       round(vsfc_report["recall"] * 100, 2),
        "F1 (%)":           round(vsfc_report["macro_f1"] * 100, 2),
        "Unc. Rate (%)":    round(vsfc_report["uncertain_rate"] * 100, 1),
        "Samples":          len(vsfc_test),
    }])
    out_csv = os.path.join(args.results_dir, "uit_vsfc_evaluation.csv")
    table_row.to_csv(out_csv, index=False)
    print(f"Summary table → {out_csv}")

    return vsfc_report


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot cross-domain evaluation on UIT-VSFC"
    )
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoints/PhoBERT_–_SR_BT_MC-Dropout_42/best_model",
        help="Path to best_model checkpoint directory from run_phobert.py",
    )
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--mc_passes",   type=int, default=20)
    parser.add_argument("--cache_dir",   default=None,
                        help="HuggingFace datasets cache directory")
    args = parser.parse_args()
    run_uit_vsfc_evaluation(args)


if __name__ == "__main__":
    main()
