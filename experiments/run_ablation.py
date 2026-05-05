"""
experiments/run_ablation.py  [v6 — entropy stratification + per-fold augmentation]
------------------------------------------------------------------------------------
Ablation study: Table 4 của paper.

THAY ĐỔI v6:
  - Dùng entropy-based stratification (Phương án B)
  - Chạy trên 1 fold đại diện (fold 0) × 1 seed (42) cho ablation
    (paper Section 4.3: "Each configuration uses a single seed for the ablation")
  - SR ablation: regenerate augmented data theo đúng rate
    (không reuse file pre-generated — đây là fix bug SR identical results)
  - BT ablation: regenerate theo từng threshold
  - Table 4 columns: SR Rate | BT Filter τ | Acc | F1 | Unc.Rate | Mean Entropy | ECE

Kiểm tra ablation bằng 1 fold giúp tiết kiệm thời gian.
Kết quả Table 4 sẽ được verify với 5-fold nếu cần.

Usage:
  python experiments/run_ablation.py --data_dir data --viwordnet path/to/viwordnet.tsv

  # Bỏ qua BT (chậm, cần GPU):
  python experiments/run_ablation.py --data_dir data --skip_bt
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.augmentation.synonym_replacement import SynonymReplacer, load_viwordnet
from src.augmentation.back_translation import BackTranslator
from src.utils.data_loader import create_5fold_splits, VLSPDataset, verify_no_data_leakage
from src.models.phobert_classifier import PhoBERTTrainer, PHOBERT_BASE
from src.evaluation.uncertainty import (
    uncertainty_report, compute_ece,
    ENTROPY_LOW_THRESHOLD, ENTROPY_HIGH_THRESHOLD,
)

RESULTS_DIR = "results/ablation"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Sinh augmented data cho ablation ─────────────────────────────────────────

def make_sr_data(train_df: pd.DataFrame, rate: float, seed: int, viwordnet_path=None):
    """
    Sinh SR augmented data với rate cụ thể.

    FIX v6: Hàm này REGENERATE augmented data mỗi lần với đúng rate —
    không reuse file pre-generated. Đây là lý do v5 cho identical results.
    """
    syn_dict = load_viwordnet(viwordnet_path)
    replacer = SynonymReplacer(rate=rate, synonym_dict=syn_dict, seed=seed)
    aug_df   = replacer.augment_dataframe(train_df)
    replacer.close()

    n_original = len(train_df)
    n_aug      = len(aug_df)
    print(f"[SR rate={rate}] Generated {n_aug} augmented mẫu từ {n_original} gốc "
          f"(ratio: {n_aug/n_original:.2f}x)")

    # Kiểm tra diversity: bao nhiêu % thực sự thay đổi
    orig_texts = set(train_df["text"].astype(str).tolist())
    changed    = aug_df[~aug_df["text"].isin(orig_texts)]
    print(f"[SR rate={rate}] Thực sự thay đổi text: {len(changed)}/{n_aug} "
          f"({len(changed)/max(n_aug,1)*100:.1f}%)")

    combined = pd.concat([train_df, aug_df], ignore_index=True)
    return combined.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_bt_data(train_df: pd.DataFrame, threshold: float, batch_size: int = 16):
    """Sinh BT augmented data với threshold cụ thể."""
    bt  = BackTranslator(sim_threshold=threshold, batch_size=batch_size)
    aug = bt.augment_dataframe(train_df)

    n_original = len(train_df)
    n_aug      = len(aug)
    print(f"[BT τ={threshold}] Accepted {n_aug}/{n_original} mẫu "
          f"(acceptance rate: {n_aug/n_original*100:.1f}%)")

    combined = pd.concat([train_df, aug[["text", "label"]]], ignore_index=True)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


# ── Single ablation run ───────────────────────────────────────────────────────

def ablation_run(
    config_label: str,
    train_df:     pd.DataFrame,
    val_df:       pd.DataFrame,
    test_df:      pd.DataFrame,
    seed:         int = 42,
) -> Dict:
    """Train và evaluate 1 ablation config trên 1 fold."""
    import torch
    from torch.utils.data import DataLoader

    trainer = PhoBERTTrainer(
        model_name = PHOBERT_BASE,
        lr=2e-5, epochs=10, batch_size=32, max_length=128,
        patience=3, seed=seed,
        output_dir=os.path.join(RESULTS_DIR, "ckpt",
                                config_label.replace(" ", "_").replace("+", "_")),
    )

    # FIX v6: segment tập trung trước để tránh double-segmentation
    from src.utils.text_preprocessing import segment_text
    train_ds = VLSPDataset([segment_text(t) for t in train_df["text"]],
                           train_df["label"], trainer.tokenizer, segment=False)
    val_ds   = VLSPDataset([segment_text(t) for t in val_df["text"]],
                           val_df["label"],   trainer.tokenizer, segment=False)
    test_ds  = VLSPDataset([segment_text(t) for t in test_df["text"]],
                           test_df["label"],  trainer.tokenizer, segment=False)

    trainer.train(train_ds, val_ds)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    trainer.model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch  = {k: v.to(trainer.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = trainer.model(**batch)["logits"]
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels)
    preds_arr  = probs_arr.argmax(axis=-1)

    # Uncertainty report với entropy stratification (Phương án B)
    report = uncertainty_report(probs_arr, labels_arr, preds_arr)
    report["ece"]    = compute_ece(probs_arr, labels_arr, preds_arr)
    report["config"] = config_label

    print(f"  [{config_label}] F1={report['macro_f1']*100:.2f}% | "
          f"Unc={report['uncertain_rate']*100:.1f}% | "
          f"Entropy={report['mean_entropy']:.3f} | "
          f"ECE={report['ece']*100:.2f}%")

    # Log stratum distribution để verify High stratum đủ lớn
    print(f"  Strata (entropy H < {ENTROPY_LOW_THRESHOLD} / < {ENTROPY_HIGH_THRESHOLD} / ≥ {ENTROPY_HIGH_THRESHOLD}):")
    for s in report["strata"]:
        print(f"    {s['stratum']}: n={s['n']} ({s.get('pct', 0):.1f}%)")

    return report


# ── Full ablation runner ──────────────────────────────────────────────────────

def run_ablation(args):
    # Dùng fold 0 cho ablation (representative fold)
    print("\nTạo 5-fold splits để lấy fold 0 cho ablation...")
    folds   = create_5fold_splits(data_dir=args.data_dir, n_splits=5)
    fold    = folds[0]
    train_df = fold["train"]
    val_df   = fold["val"]
    test_df  = fold["test"]

    print(f"\n[Ablation] Dùng Fold 1 (fold_idx=0): "
          f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    verify_no_data_leakage(train_df, test_df)

    rows = []

    # ── A) SR Rate Ablation ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("ABLATION A: Synonym Replacement Rate")
    print(f"[Entropy thresholds] Low<{ENTROPY_LOW_THRESHOLD} | High≥{ENTROPY_HIGH_THRESHOLD}")
    print("=" * 55)

    for rate in [0.10, 0.15, 0.20]:
        label    = f"PhoBERT+SR (rate={rate})"
        print(f"\n[Ablation] {label}")
        combined = make_sr_data(train_df, rate, seed=42,
                                viwordnet_path=args.viwordnet)
        report   = ablation_run(label, combined, val_df, test_df)
        rows.append({
            "Configuration":  label,
            "SR Rate":        str(rate),
            "BT Filter τ":    "—",
            "Accuracy (%)":   round(report["accuracy"] * 100, 2),
            "F1-Score (%)":   round(report["macro_f1"] * 100, 2),
            "Unc. Rate (%)":  round(report["uncertain_rate"] * 100, 1),
            "Mean Entropy":   round(report["mean_entropy"], 3),
            "ECE (%)":        round(report["ece"] * 100, 2),
        })

    # ── B) BT Threshold Ablation ──────────────────────────────────────────────
    if not args.skip_bt:
        print("\n" + "=" * 55)
        print("ABLATION B: Back-Translation Quality Threshold")
        print("=" * 55)

        for threshold in [0.75, 0.85, 0.90]:
            label    = f"PhoBERT+BT (τ={threshold})"
            print(f"\n[Ablation] {label}")
            combined = make_bt_data(train_df, threshold, batch_size=args.bt_batch_size)
            report   = ablation_run(label, combined, val_df, test_df)
            rows.append({
                "Configuration":  label,
                "SR Rate":        "—",
                "BT Filter τ":    str(threshold),
                "Accuracy (%)":   round(report["accuracy"] * 100, 2),
                "F1-Score (%)":   round(report["macro_f1"] * 100, 2),
                "Unc. Rate (%)":  round(report["uncertain_rate"] * 100, 1),
                "Mean Entropy":   round(report["mean_entropy"], 3),
                "ECE (%)":        round(report["ece"] * 100, 2),
            })

    # ── Full config: SR(0.15) + BT(0.85) ─────────────────────────────────────
    print(f"\n[Ablation] Full config: PhoBERT + SR(0.15) + BT(0.75) [independent]")

    sr_only_df  = make_sr_data(train_df, 0.15, seed=42, viwordnet_path=args.viwordnet)
    sr_aug_only = sr_only_df.iloc[len(train_df):].copy()   # chỉ lấy phần augmented

    if not args.skip_bt:
        bt_from_orig = make_bt_data(train_df, 0.75, batch_size=args.bt_batch_size)  # FIX v6: 0.85→0.75
        bt_aug_only  = bt_from_orig.iloc[len(train_df):].copy()
        combined_all = pd.concat(
            [train_df, sr_aug_only, bt_aug_only], ignore_index=True
        ).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        combined_all = sr_only_df

    report = ablation_run("PhoBERT+SR+BT (full)", combined_all, val_df, test_df)
    rows.append({
        "Configuration":  "PhoBERT + SR + BT (Full)",
        "SR Rate":        "0.15",
        "BT Filter τ":    "0.75",
        "Accuracy (%)":   round(report["accuracy"] * 100, 2),
        "F1-Score (%)":   round(report["macro_f1"] * 100, 2),
        "Unc. Rate (%)":  round(report["uncertain_rate"] * 100, 1),
        "Mean Entropy":   round(report["mean_entropy"], 3),
        "ECE (%)":        round(report["ece"] * 100, 2),
    })

    # ── Lưu Table 4 ──────────────────────────────────────────────────────────
    table4 = pd.DataFrame(rows)
    table4.to_csv(os.path.join(RESULTS_DIR, "table4_ablation.csv"), index=False)
    print("\n=== Table 4 (Ablation Study) ===")
    print(f"[Entropy thresholds] Low<{ENTROPY_LOW_THRESHOLD} | "
          f"Medium [{ENTROPY_LOW_THRESHOLD},{ENTROPY_HIGH_THRESHOLD}) | "
          f"High≥{ENTROPY_HIGH_THRESHOLD}")
    print(table4.to_string(index=False))

    with open(os.path.join(RESULTS_DIR, "ablation_reports.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: SR rate và BT threshold (entropy stratification)"
    )
    parser.add_argument("--data_dir",      default="data")
    parser.add_argument("--viwordnet",     default=None,
                        help="Đường dẫn tới ViWordNet .tsv file (BẮT BUỘC cho SR diversity)")
    parser.add_argument("--skip_bt",       action="store_true",
                        help="Bỏ qua BT ablation (chậm, cần GPU mạnh)")
    parser.add_argument("--bt_batch_size", type=int, default=16)
    args = parser.parse_args()

    if args.viwordnet is None:
        print("[WARNING] --viwordnet không được chỉ định. "
              "Dùng built-in SYNONYM_DICT (~30 từ) → SR diversity thấp, "
              "ablation rates có thể cho kết quả tương tự nhau.")

    run_ablation(args)


if __name__ == "__main__":
    main()
