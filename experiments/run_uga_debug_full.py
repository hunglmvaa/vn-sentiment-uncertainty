"""
experiments/run_uga.py  [v6-debug — entropy-based thresholds cho D_high/D_med]
-------------------------------------------------------------------------
UGA (Uncertainty-Guided Augmentation) — Algorithm 1.

THAY ĐỔI v6-debug:
  - alpha/beta là ngưỡng ENTROPY (Phương án B):
      alpha = 0.80  → D_high: entropy ≥ 0.80  (~max_softmax < 0.60)
      beta  = 0.40  → D_med : entropy ∈ [0.40, 0.80)
  - Dùng entropy từ MC-Dropout ensemble để partition training set
  - Pre-flight diagnostics C1–C5 chạy tự động sau Round 1
  - --force flag để override nếu diagnostics không pass
  - FIX UTF-8 cho Windows / PowerShell
  - FIX JSON serialization cho numpy.bool_ / ndarray / scalar
  - DEBUG dataset composition:
      + len(current_train), len(aug_high), len(aug_med), len(d_augmented)
      + label distribution của current_train / aug_high / aug_med / d_augmented
      + len(train_ds), len(val_ds), len(test_ds)
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

if hasattr(sys.stdout, "reconfigure") and sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure") and sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import (
    create_5fold_splits, VLSPDataset, verify_no_data_leakage
)
from src.models.phobert_classifier import PhoBERTTrainer, PHOBERT_BASE
from src.augmentation.synonym_replacement import SynonymReplacer, load_viwordnet
from src.augmentation.back_translation import BackTranslator
from src.evaluation.uncertainty import (
    uncertainty_report, compute_ece, diagnose_uga_conditions,
    print_uga_diagnostics, compute_entropy,
    ENTROPY_LOW_THRESHOLD, ENTROPY_HIGH_THRESHOLD,
)

RESULTS_DIR = "results/uga"
os.makedirs(RESULTS_DIR, exist_ok=True)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def partition_by_entropy(
    train_df: pd.DataFrame,
    ensemble_probs: np.ndarray,
    alpha: float,
    beta: float,
) -> Dict[str, pd.DataFrame]:
    entropy = compute_entropy(ensemble_probs)

    mask_high = entropy >= alpha
    mask_med = (entropy >= beta) & (entropy < alpha)
    mask_low = entropy < beta

    d_high = train_df[mask_high].reset_index(drop=True)
    d_med = train_df[mask_med].reset_index(drop=True)
    d_low = train_df[mask_low].reset_index(drop=True)

    n = len(train_df)
    print(f"\n[UGA Partition — entropy thresholds α={alpha}, β={beta}]")
    print(f"  D_high (H≥{alpha}): {len(d_high)} mẫu ({len(d_high)/n*100:.1f}%)")
    print(f"  D_med  ({beta}≤H<{alpha}): {len(d_med)} mẫu ({len(d_med)/n*100:.1f}%)")
    print(f"  D_low  (H<{beta}): {len(d_low)} mẫu ({len(d_low)/n*100:.1f}%)")

    return {"high": d_high, "med": d_med, "low": d_low}


def ensemble_mc_probs(
    train_df: pd.DataFrame,
    ensemble_seeds: List[int],
    data_dir: str,
    fold: Dict,
    mc_passes: int = 20,
    output_dir: str = "results/uga/ensemble_ckpt",
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    val_df = fold["val"]
    train_df_clean = fold["train"]
    all_seed_probs = []

    for seed in ensemble_seeds:
        print(f"\n  [Ensemble] Training seed {seed}...")
        trainer = PhoBERTTrainer(
            model_name=PHOBERT_BASE,
            lr=2e-5,
            epochs=5,
            batch_size=32,
            max_length=128,
            patience=3,
            seed=seed,
            output_dir=os.path.join(output_dir, f"seed_{seed}"),
        )

        train_ds = VLSPDataset(train_df_clean["text"], train_df_clean["label"], trainer.tokenizer)
        val_ds = VLSPDataset(val_df["text"], val_df["label"], trainer.tokenizer)
        trainer.train(train_ds, val_ds)

        score_ds = VLSPDataset(train_df["text"], train_df["label"], trainer.tokenizer)
        score_loader = DataLoader(score_ds, batch_size=32, shuffle=False)
        result = trainer.predict_with_uncertainty(score_loader, mc_passes=mc_passes)
        all_seed_probs.append(result["mean_probs"])

    ensemble_probs = np.array(all_seed_probs).mean(axis=0)
    print(f"\n[Ensemble] {len(ensemble_seeds)} seeds × {mc_passes} passes → "
          f"ensemble probs shape: {ensemble_probs.shape}")
    return ensemble_probs


def augment_partition(
    subset_df: pd.DataFrame,
    factor: int,
    bt_threshold: float,
    viwordnet_path: Optional[str],
    sr_seed: int = 42,
    bt_batch_size: int = 16,
    label: str = "",
) -> pd.DataFrame:
    if len(subset_df) == 0:
        print(f"[UGA Aug] {label} rỗng — bỏ qua.")
        return pd.DataFrame(columns=["text", "label"])

    aug_frames = [subset_df.copy()]

    n_sr = factor // 2
    syn_dict = load_viwordnet(viwordnet_path)
    for i in range(n_sr):
        replacer = SynonymReplacer(rate=0.15, synonym_dict=syn_dict, seed=sr_seed + i)
        sr_df = replacer.augment_dataframe(subset_df)
        replacer.close()
        aug_frames.append(sr_df)
        print(f"  [SR ×{i+1}/{n_sr}] +{len(sr_df)} mẫu")

    n_bt = factor // 2
    for i in range(n_bt):
        bt = BackTranslator(sim_threshold=bt_threshold, batch_size=bt_batch_size)
        bt_df = bt.augment_dataframe(subset_df)
        aug_frames.append(bt_df[["text", "label"]])
        print(f"  [BT ×{i+1}/{n_bt}, τ={bt_threshold}] +{len(bt_df)} mẫu accepted")

    combined = pd.concat(aug_frames, ignore_index=True)

    target_n = len(subset_df) * factor
    if len(combined) > target_n:
        combined = combined.groupby("label", group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(target_n * len(x) / len(combined))),
                random_state=42
            )
        ).reset_index(drop=True)

    print(f"  [{label} ×{factor}] Tổng: {len(combined)} mẫu "
          f"(từ {len(subset_df)} gốc, target={target_n})")
    return combined


def _label_dist(df: pd.DataFrame) -> Dict[int, int]:
    if df is None or len(df) == 0 or "label" not in df.columns:
        return {}
    vc = df["label"].value_counts().sort_index()
    return {int(k): int(v) for k, v in vc.items()}


def run_uga(args):
    print("\nTạo 5-fold splits...")
    folds = create_5fold_splits(data_dir=args.data_dir, n_splits=5)
    fold = folds[0]

    train_df = fold["train"].copy()
    val_df = fold["val"]
    test_df = fold["test"]

    print(f"\n[UGA] Dùng Fold 1: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    verify_no_data_leakage(train_df, test_df)

    round_history = []
    current_train = train_df.copy()

    for round_num in range(1, args.max_rounds + 1):
        print(f"\n{'='*55}")
        print(f"UGA ROUND {round_num}/{args.max_rounds}")
        print(f"  Entropy thresholds: α={args.alpha}, β={args.beta}")
        print(f"{'='*55}")

        print(f"\n[Round {round_num}] Tính ensemble entropy trên training set...")
        ensemble_probs = ensemble_mc_probs(
            train_df=current_train,
            ensemble_seeds=args.ensemble_seeds,
            data_dir=args.data_dir,
            fold=fold,
            mc_passes=args.mc_passes,
            output_dir=os.path.join(RESULTS_DIR, f"round{round_num}_ensemble"),
        )

        partitions = partition_by_entropy(
            train_df=current_train,
            ensemble_probs=ensemble_probs,
            alpha=args.alpha,
            beta=args.beta,
        )

        diag = diagnose_uga_conditions(
            baseline_probs=ensemble_probs,
            baseline_labels=current_train["label"].values,
            baseline_preds=ensemble_probs.argmax(axis=-1),
            train_probs=ensemble_probs,
            train_labels=current_train["label"].values,
            alpha=args.alpha,
            beta=args.beta,
        )
        print_uga_diagnostics(diag)

        diag_path = os.path.join(RESULTS_DIR, f"uga_condition_diagnostics_round{round_num}.json")
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(diag), f, indent=2, ensure_ascii=False)
        print(f"[Diagnostics] Lưu vào {diag_path}")

        dhigh_frac = diag.get("C4_dhigh_frac", 0) / 100

        if not diag["uga_recommended"] and not args.force:
            print(f"\n[UGA] Conditions không pass sau Round {round_num}.")
            print("      Dùng --force để tiếp tục bất chấp diagnostics.")
            print(f"      Xem {diag_path} để biết chi tiết.")
            break

        if dhigh_frac <= 0.02:
            print(f"\n[UGA] Stopping criterion met: D_high = {dhigh_frac*100:.1f}% ≤ 2%.")
            break

        print(f"\n[Round {round_num}] Augmenting D_high (×4) và D_med (×2)...")

        aug_high = augment_partition(
            subset_df=partitions["high"],
            factor=4,
            bt_threshold=0.90,
            viwordnet_path=args.viwordnet,
            label="D_high",
            bt_batch_size=args.bt_batch_size,
        )

        aug_med = augment_partition(
            subset_df=partitions["med"],
            factor=2,
            bt_threshold=0.75,
            viwordnet_path=args.viwordnet,
            label="D_med",
            bt_batch_size=args.bt_batch_size,
        )

        d_augmented = pd.concat(
            [current_train, aug_high, aug_med], ignore_index=True
        ).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n[Round {round_num}] D_train = {len(current_train)} → "
              f"D_augmented = {len(d_augmented)} mẫu "
              f"(+{len(d_augmented)-len(current_train)} augmented)")

        print("\n[DEBUG] ===== Dataset composition debug =====")
        print(f"[DEBUG] len(current_train) = {len(current_train)}")
        print(f"[DEBUG] len(aug_high)     = {len(aug_high)}")
        print(f"[DEBUG] len(aug_med)      = {len(aug_med)}")
        print(f"[DEBUG] len(d_augmented)  = {len(d_augmented)}")
        print(f"[DEBUG] current_train label distribution = {_label_dist(current_train)}")
        print(f"[DEBUG] aug_high label distribution     = {_label_dist(aug_high)}")
        print(f"[DEBUG] aug_med label distribution      = {_label_dist(aug_med)}")
        print(f"[DEBUG] d_augmented label distribution  = {_label_dist(d_augmented)}")
        print("[DEBUG] ====================================")

        print(f"\n[Round {round_num}] Re-training PhoBERT trên D_augmented...")
        import torch
        from torch.utils.data import DataLoader

        trainer = PhoBERTTrainer(
            model_name=PHOBERT_BASE,
            lr=2e-5,
            epochs=10,
            batch_size=32,
            max_length=128,
            patience=3,
            seed=42,
            output_dir=os.path.join(RESULTS_DIR, f"round{round_num}_model"),
        )

        train_ds = VLSPDataset(d_augmented["text"], d_augmented["label"], trainer.tokenizer)
        val_ds = VLSPDataset(val_df["text"], val_df["label"], trainer.tokenizer)
        test_ds = VLSPDataset(test_df["text"], test_df["label"], trainer.tokenizer)

        print(f"[DEBUG] len(train_ds) = {len(train_ds)}")
        print(f"[DEBUG] len(val_ds)   = {len(val_ds)}")
        print(f"[DEBUG] len(test_ds)  = {len(test_ds)}")

        train_result = trainer.train(train_ds, val_ds)
        print(f"  Best val F1: {train_result['best_val_f1']:.4f}")

        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        uga_result = trainer.predict_with_uncertainty(test_loader, mc_passes=args.mc_passes)

        test_report = uncertainty_report(
            probs=np.array(uga_result["mean_probs"]),
            labels=np.array(uga_result["labels"]),
            predictions=np.array(uga_result["predictions"]),
        )
        test_report["ece"] = compute_ece(
            np.array(uga_result["mean_probs"]),
            np.array(uga_result["labels"]),
            np.array(uga_result["predictions"]),
        )

        round_summary = {
            "round": round_num,
            "dhigh_frac": diag.get("C4_dhigh_frac"),
            "n_augmented": len(d_augmented),
            "f1": test_report["macro_f1"],
            "accuracy": test_report["accuracy"],
            "uncertain_rate": test_report["uncertain_rate"],
            "mean_entropy": test_report["mean_entropy"],
            "ece": test_report["ece"],
        }
        round_history.append(round_summary)

        print(f"\n[Round {round_num} Results]")
        print(f"  F1           = {test_report['macro_f1']*100:.2f}%")
        print(f"  Accuracy     = {test_report['accuracy']*100:.2f}%")
        print(f"  Unc. Rate    = {test_report['uncertain_rate']*100:.1f}%")
        print(f"  Mean Entropy = {test_report['mean_entropy']:.3f}")
        print(f"  ECE          = {test_report['ece']*100:.2f}%")
        print(f"\n  Strata (entropy H<{args.beta}/{args.beta}-{args.alpha}/≥{args.alpha}):")
        for s in test_report["strata"]:
            if s.get("accuracy") is not None:
                print(f"    {s['stratum']}: n={s['n']} ({s.get('pct',0):.1f}%) | "
                      f"acc={s.get('accuracy',0)*100:.1f}%")
            else:
                print(f"    {s['stratum']}: n=0")

        current_train = d_augmented.copy()

    final_report = {
        "alpha": args.alpha,
        "beta": args.beta,
        "stratification": "entropy (Phương án B)",
        "thresholds_note": (
            f"D_high: entropy ≥ {args.alpha} (~max_softmax < 0.60); "
            f"D_med: entropy [{args.beta}, {args.alpha}); "
            f"D_low: entropy < {args.beta} (~max_softmax > 0.82)"
        ),
        "rounds": round_history,
        "total_rounds": len(round_history),
    }

    final_path = os.path.join(RESULTS_DIR, "uga_final_report.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(final_report), f, indent=2, ensure_ascii=False)
    print(f"\n[UGA Complete] Final report: {final_path}")

    print("\n=== UGA Round History ===")
    for r in round_history:
        print(f"  Round {r['round']}: F1={r['f1']*100:.2f}% | "
              f"D_high={r['dhigh_frac']:.1f}% | "
              f"Entropy={r['mean_entropy']:.3f} | "
              f"ECE={r['ece']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="UGA: Uncertainty-Guided Augmentation (entropy-based, Phương án B)"
    )
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--max_rounds", type=int, default=3)
    parser.add_argument("--mc_passes", type=int, default=20)
    parser.add_argument("--ensemble_seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--alpha", type=float, default=ENTROPY_HIGH_THRESHOLD,
                        help=f"Entropy threshold cho D_high (default: {ENTROPY_HIGH_THRESHOLD})")
    parser.add_argument("--beta", type=float, default=ENTROPY_LOW_THRESHOLD,
                        help=f"Entropy threshold cho D_med  (default: {ENTROPY_LOW_THRESHOLD})")
    parser.add_argument("--viwordnet", default=None)
    parser.add_argument("--bt_batch_size", type=int, default=16)
    parser.add_argument("--force", action="store_true",
                        help="Override diagnostics và tiếp tục UGA bất kể C-conditions")
    args = parser.parse_args()

    print(f"\nUGA start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Stratification: ENTROPY (Phương án B)")
    print(f"α={args.alpha} (D_high ≥ {args.alpha}) | β={args.beta} (D_med ≥ {args.beta})")
    run_uga(args)
    print(f"\nUGA end: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
