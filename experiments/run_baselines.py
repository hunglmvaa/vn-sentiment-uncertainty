"""
experiments/run_baselines.py
-----------------------------
Train and evaluate SVM + BiLSTM baselines on VLSP 2016.
Generates Table 3 rows for classical methods.
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.data_loader import load_split
from src.models.baseline_models import SVMClassifier, BiLSTMTrainer

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_svm(data_dir: str, seeds: list) -> dict:
    val_df  = load_split(os.path.join(data_dir, "raw"), "val")
    test_df = load_split(os.path.join(data_dir, "raw"), "test")

    seed_reports = []
    for seed in seeds:
        train_df = load_split(os.path.join(data_dir, "raw"), "train")

        clf = SVMClassifier(seed=seed)
        clf.train(train_df["text"].tolist(), train_df["label"].tolist())

        train_metrics = clf.evaluate(train_df["text"].tolist(), train_df["label"].tolist())
        val_metrics   = clf.evaluate(val_df["text"].tolist(),   val_df["label"].tolist())
        test_metrics  = clf.evaluate(test_df["text"].tolist(),  test_df["label"].tolist())

        print(f"\n[SVM seed={seed}]")
        print(f"  Train F1 : {train_metrics['macro_f1']*100:.2f}")
        print(f"  Val   F1 : {val_metrics  ['macro_f1']*100:.2f}")
        print(f"  Test  F1 : {test_metrics ['macro_f1']*100:.2f}")

        report = {"seed": seed, **test_metrics}
        seed_reports.append(report)

        clf.save(os.path.join(RESULTS_DIR, f"svm_seed{seed}.pkl"))

    # Aggregate
    f1s = [r["macro_f1"] for r in seed_reports]
    print(f"\n[SVM] Mean F1: {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")

    with open(os.path.join(RESULTS_DIR, "svm_results.json"), "w") as f:
        json.dump(seed_reports, f, indent=2)
    return {"mean_f1": np.mean(f1s), "std_f1": np.std(f1s), "reports": seed_reports}


def run_bilstm(data_dir: str, seeds: list, emb_path: str = None) -> dict:
    val_df  = load_split(os.path.join(data_dir, "raw"), "val")
    test_df = load_split(os.path.join(data_dir, "raw"), "test")

    seed_reports = []
    for seed in seeds:
        train_df = load_split(os.path.join(data_dir, "raw"), "train")

        trainer = BiLSTMTrainer(
            embed_dim   = 100,
            hidden_size = 128,
            num_layers  = 2,
            lr          = 1e-3,
            epochs      = 20,
            batch_size  = 64,
            patience    = 5,
            seed        = seed,
            output_dir  = os.path.join(RESULTS_DIR, f"bilstm_seed{seed}"),
            pretrained_emb_path = emb_path,
        )

        trainer.train(
            train_df["text"].tolist(), train_df["label"].tolist(),
            val_df["text"].tolist(),   val_df["label"].tolist(),
        )

        test_metrics = trainer.evaluate(
            test_df["text"].tolist(), test_df["label"].tolist()
        )

        print(f"\n[BiLSTM seed={seed}]")
        print(f"  Test Acc : {test_metrics['accuracy']*100:.2f}")
        print(f"  Test F1  : {test_metrics['macro_f1']*100:.2f}")

        report = {"seed": seed, **test_metrics}
        seed_reports.append(report)

    f1s = [r["macro_f1"] for r in seed_reports]
    print(f"\n[BiLSTM] Mean F1: {np.mean(f1s)*100:.2f} ± {np.std(f1s)*100:.2f}")

    with open(os.path.join(RESULTS_DIR, "bilstm_results.json"), "w") as f:
        json.dump(seed_reports, f, indent=2)
    return {"mean_f1": np.mean(f1s), "std_f1": np.std(f1s), "reports": seed_reports}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--emb_path", default=None,
                        help="Path to pre-trained Word2Vec .txt embeddings (optional)")
    args = parser.parse_args()

    print("\n" + "="*50)
    print("Running SVM baseline...")
    print("="*50)
    svm_result = run_svm(args.data_dir, args.seeds)

    print("\n" + "="*50)
    print("Running BiLSTM baseline...")
    print("="*50)
    bilstm_result = run_bilstm(args.data_dir, args.seeds, args.emb_path)

    print("\n" + "="*50)
    print("BASELINE SUMMARY")
    print("="*50)
    print(f"SVM    F1: {svm_result   ['mean_f1']*100:.2f} ± {svm_result   ['std_f1']*100:.2f}")
    print(f"BiLSTM F1: {bilstm_result['mean_f1']*100:.2f} ± {bilstm_result['std_f1']*100:.2f}")


if __name__ == "__main__":
    main()
