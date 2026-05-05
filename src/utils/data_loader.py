"""
utils/data_loader.py  [v7 â€” fold-safe augmentation]
---------------------------------------------------
Dataset loading, splitting, vÃ  chuáº©n bá»‹ cho VLSP 2016.

THAY Äá»”I v7:
  - Giá»¯ 5-fold stratified CV
  - Viáº¿t láº¡i load_augmented_for_fold() Ä‘á»ƒ chá»‘ng leakage Ä‘Ãºng theo fold
  - Há»— trá»£ augmented files cÃ³ source_id hoáº·c source_text
  - KhÃ´ng cÃ²n random sample 80% augmented file toÃ n cá»¥c
  - load_full_dataset() khÃ´ng drop_duplicates toÃ n cá»¥c Ä‘á»ƒ trÃ¡nh lÃ m thay Ä‘á»-i corpus gá»‘c

CSV format tá»‘i thiá»ƒu:
  cá»™t [text, label]
  label: 0=negative, 1=neutral, 2=positive (hoáº·c string)

Augmented CSV format khuyáº¿n nghá»‹:
  [source_id, source_text, text, label, ...]
hoáº·c tá»‘i thiá»ƒu:
  [source_text, text, label, ...]
"""

import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset

from src.utils.text_preprocessing import segment_text


LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3


# â”€â”€ Dataset class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class VLSPDataset(Dataset):
    """
    PyTorch Dataset cho VLSP 2016 Vietnamese Sentiment.

    Parameters
    ----------
    texts     : iterable of str
    labels    : iterable of int
    tokenizer : HuggingFace tokenizer (PhoBERT / mBERT)
    max_length: int, default 128
    segment   : bool, default True
        Khi True, má»—i text Ä‘Æ°á»£c segment báº±ng underthesea trÆ°á»›c khi tokenize.
        Set False náº¿u text Ä‘Ã£ Ä‘Æ°á»£c segment sáºµn.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128, segment=True):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tok = tokenizer
        self.max_len = max_length
        self.segment = segment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.segment:
            text = segment_text(text)

        enc = self.tok(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# â”€â”€ CSV loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV vá»›i cá»™t tá»‘i thiá»ƒu [text, label].
    Normalize label sang int.
    Giá»¯ láº¡i cÃ¡c cá»™t phá»¥ nhÆ° source_id/source_text náº¿u cÃ³.
    """
    df = pd.read_csv(path)

    assert "text" in df.columns and "label" in df.columns, (
        f"CSV pháº£i cÃ³ cá»™t 'text' vÃ  'label'. TÃ¬m tháº¥y: {df.columns.tolist()}"
    )

    if df["label"].dtype == object:
        df["label"] = df["label"].map(LABEL2ID)

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    assert df["label"].isin([0, 1, 2]).all(), (
        f"Label pháº£i lÃ  0/1/2. TÃ¬m tháº¥y: {df['label'].unique()}"
    )

    if "source_text" in df.columns:
        df["source_text"] = df["source_text"].astype(str)

    if "source_id" in df.columns:
        df["source_id"] = df["source_id"].astype(str)

    return df.reset_index(drop=True)


def load_split(data_dir: str, split: str) -> pd.DataFrame:
    """Load train / val / test CSV tá»« data_dir."""
    path = os.path.join(data_dir, f"{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"KhÃ´ng tÃ¬m tháº¥y: {path}")

    df = load_csv(path)
    print(
        f"[{split}] Loaded {len(df)} samples | "
        f"pos={sum(df.label == 2)}, neu={sum(df.label == 1)}, neg={sum(df.label == 0)}"
    )
    return df


def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load toÃ n bá»™ VLSP dataset báº±ng cÃ¡ch ghÃ©p train + val + test.
    DÃ¹ng cho 5-fold CV.

    Náº¿u chá»‰ cÃ³ train.csv, load train.csv.
    Náº¿u cÃ³ full.csv hoáº·c vlsp2016.csv thÃ¬ dÃ¹ng trá»±c tiáº¿p.
    """
    frames = []

    for split in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split}.csv")
        if os.path.exists(path):
            frames.append(load_csv(path))

    if not frames:
        for name in ["full.csv", "vlsp2016.csv", "vlsp2016_full.csv"]:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                df = load_csv(path)
                print(f"[full] Loaded {len(df)} samples tá»« {name}")
                return df

        raise FileNotFoundError(
            f"KhÃ´ng tÃ¬m tháº¥y data trong {data_dir}. "
            f"Cáº§n cÃ³ train.csv (+ val.csv + test.csv) hoáº·c full.csv"
        )

    full_df = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    print(
        f"[full] GhÃ©p {len(frames)} splits â†’ {len(full_df)} máº«u | "
        f"pos={sum(full_df.label == 2)}, neu={sum(full_df.label == 1)}, neg={sum(full_df.label == 0)}"
    )
    return full_df


# â”€â”€ 5-fold CV â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def create_5fold_splits(
    data_dir: str,
    n_splits: int = 5,
    val_fraction: float = 0.10,
    random_state: int = 42,
) -> List[Dict]:
    """
    Táº¡o stratified k-fold splits.

    Returns
    -------
    List[Dict] gá»“m:
      {
        "fold": int,
        "train": pd.DataFrame,
        "val": pd.DataFrame,
        "test": pd.DataFrame,
        "train_idx": np.ndarray,
        "test_idx": np.ndarray,
      }
    """
    raw_dir = os.path.join(data_dir, "raw")
    full_df = load_full_dataset(raw_dir)

    total = len(full_df)
    expected_test = total // n_splits

    print(f"\n[5-fold CV] Tá»-ng {total} máº«u â†’ {n_splits} folds Ã— ~{expected_test} test máº«u")
    print(
        f"[5-fold CV] Má»—i training fold: ~{total - expected_test} máº«u â†’ "
        f"val ~{int((total - expected_test) * val_fraction)}, "
        f"train ~{int((total - expected_test) * (1 - val_fraction))}"
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []

    def class_dist(df: pd.DataFrame) -> Dict[str, int]:
        counts = df["label"].value_counts().sort_index()
        return {ID2LABEL[i]: int(counts.get(i, 0)) for i in range(3)}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(full_df, full_df["label"])):
        train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
        test_df = full_df.iloc[test_idx].reset_index(drop=True)

        train_df, val_df = train_test_split(
            train_fold_df,
            test_size=val_fraction,
            stratify=train_fold_df["label"],
            random_state=random_state,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        print(f"\n  Fold {fold_idx + 1}/{n_splits}:")
        print(f"    train={len(train_df)} {class_dist(train_df)}")
        print(f"    val  ={len(val_df)}   {class_dist(val_df)}")
        print(f"    test ={len(test_df)}  {class_dist(test_df)}")

        splits.append(
            {
                "fold": fold_idx,
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )

    return splits


def _resolve_augmented_path(data_dir: str, canonical_name: str, variants: Optional[List[str]] = None) -> Optional[str]:
    """Resolve canonical augmented file path with safe fallbacks.

    Search order:
      1. data_dir/augmented/<canonical_name>
      2. parent_of_data_dir/augmented/<canonical_name>
      3. variant filenames in both locations
    """
    variants = variants or []
    search_roots = [os.path.join(data_dir, "augmented")]
    parent_aug = os.path.join(os.path.dirname(data_dir), "augmented")
    if parent_aug not in search_roots:
        search_roots.append(parent_aug)

    candidate_names = [canonical_name] + [v for v in variants if v != canonical_name]
    for root in search_roots:
        for name in candidate_names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path
    return None


# â”€â”€ Augmented data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def load_augmented_train(data_dir: str, config: str) -> pd.DataFrame:
    """
    Load augmented training data cho pipeline train/val/test cá»‘ Ä‘á»‹nh.

    config:
        'baseline' â€“ original train only
        'sr'       â€“ original + synonym-replaced
        'bt'       â€“ original + back-translated
        'sr+bt'    â€“ original + SR + BT
    """
    orig = load_csv(os.path.join(data_dir, "raw", "train.csv"))
    if config == "baseline":
        return orig

    frames = [orig]

    if "sr" in config:
        sr_path = _resolve_augmented_path(data_dir, "train_sr.csv", variants=["train_sr_viwordnet.tsv.csv", "train_sr_final.csv", "train_sr_vinetjson.csv"])
        if sr_path:
            sr_df = load_csv(sr_path)
            frames.append(sr_df[["text", "label"]])
        else:
            print("[WARNING] SR file khÃ´ng tÃ¬m tháº¥y trong augmented roots.")

    if "bt" in config:
        bt_path = _resolve_augmented_path(data_dir, "train_bt.csv", variants=["train_bt_final.csv"])
        if bt_path:
            bt_df = load_csv(bt_path)
            frames.append(bt_df[["text", "label"]])
        else:
            print("[WARNING] BT file khÃ´ng tÃ¬m tháº¥y trong augmented roots.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"]).sample(
        frac=1.0, random_state=42
    ).reset_index(drop=True)

    print(f"[train/{config}] Tá»-ng sau augmentation: {len(combined)} máº«u")
    return combined


def load_augmented_for_fold(
    data_dir: str,
    train_df: pd.DataFrame,
    aug_config: str,
    fold_idx: int = 0,
    remove_identity_aug: bool = True,
    deduplicate_on_text: bool = True,
) -> pd.DataFrame:
    """
    Load augmented data CHO Má»˜T FOLD Cá»¤ THá»‚, chá»‘ng leakage Ä‘Ãºng theo fold.

    Logic:
    - Chá»‰ giá»¯ augmented rows cÃ³ source_id/source_text thuá»™c train_df cá»§a fold hiá»‡n táº¡i.
    - KhÃ´ng cÃ²n random sample 80% augmented file toÃ n cá»¥c.
    - Náº¿u augmented file khÃ´ng cÃ³ source_id/source_text thÃ¬ bÃ¡o lá»—i.

    YÃªu cáº§u augmented CSV:
    - text, label
    - vÃ  thÃªm source_id hoáº·c source_text

    Parameters
    ----------
    data_dir : str
        Root data directory.
    train_df : pd.DataFrame
        Training split gá»‘c cá»§a fold hiá»‡n táº¡i.
    aug_config : str
        'baseline', 'sr', 'bt', 'sr+bt'
    fold_idx : int
        Chá»‰ dÃ¹ng cho log.
    remove_identity_aug : bool
        Bá» augmented rows náº¿u text giá»‘ng há»‡t raw train text.
    deduplicate_on_text : bool
        Drop duplicate theo text sau khi merge.

    Returns
    -------
    pd.DataFrame
        DataFrame Ä‘Ã£ gá»™p: original train + fold-safe augmentation.
    """
    if aug_config == "baseline":
        return train_df.copy().reset_index(drop=True)

    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError("train_df pháº£i cÃ³ cá»™t ['text', 'label'].")

    train_df = train_df.copy()
    train_df["text"] = train_df["text"].astype(str)
    raw_train_texts = set(train_df["text"].tolist())

    has_train_source_id = "source_id" in train_df.columns
    if has_train_source_id:
        train_df["source_id"] = train_df["source_id"].astype(str)
        train_source_ids = set(train_df["source_id"].tolist())
    else:
        train_source_ids = set()

    def _load_and_filter_aug(file_path: str, tag: str) -> pd.DataFrame:
        if not os.path.exists(file_path):
            print(f"[WARNING][Fold {fold_idx}] {tag} file khÃ´ng tÃ¬m tháº¥y: {file_path}")
            return pd.DataFrame(columns=["text", "label"])

        aug_df = load_csv(file_path)

        if "source_id" in aug_df.columns and has_train_source_id:
            aug_df["source_id"] = aug_df["source_id"].astype(str)
            filtered = aug_df[aug_df["source_id"].isin(train_source_ids)].copy()
            match_mode = "source_id"
        elif "source_text" in aug_df.columns:
            aug_df["source_text"] = aug_df["source_text"].astype(str)
            filtered = aug_df[aug_df["source_text"].isin(raw_train_texts)].copy()
            match_mode = "source_text"
        else:
            raise ValueError(
                f"{tag} file pháº£i chá»©a 'source_id' hoáº·c 'source_text' "
                f"Ä‘á»ƒ lá»c augmentation theo fold an toÃ n."
            )

        filtered = filtered.dropna(subset=["text", "label"]).copy()
        filtered["text"] = filtered["text"].astype(str)
        filtered["label"] = filtered["label"].astype(int)

        if remove_identity_aug:
            filtered = filtered[~filtered["text"].isin(raw_train_texts)].copy()

        if deduplicate_on_text and not filtered.empty:
            filtered = filtered.drop_duplicates(subset=["text"]).copy()

        print(
            f"[{tag}/Fold {fold_idx}] Giá»¯ {len(filtered)}/{len(aug_df)} samples "
            f"(matched by {match_mode})"
        )

        return filtered

    frames = [train_df[["text", "label"]].copy()]

    if "sr" in aug_config:
        sr_path = _resolve_augmented_path(data_dir, "train_sr.csv", variants=["train_sr_viwordnet.tsv.csv", "train_sr_final.csv", "train_sr_vinetjson.csv"])
        sr_df = _load_and_filter_aug(sr_path, "SR") if sr_path else pd.DataFrame(columns=["text", "label"])
        if not sr_df.empty:
            frames.append(sr_df[["text", "label"]])

    if "bt" in aug_config:
        bt_path = _resolve_augmented_path(data_dir, "train_bt.csv", variants=["train_bt_final.csv"])
        bt_df = _load_and_filter_aug(bt_path, "BT") if bt_path else pd.DataFrame(columns=["text", "label"])
        if not bt_df.empty:
            frames.append(bt_df[["text", "label"]])

    combined = pd.concat(frames, ignore_index=True)

    if deduplicate_on_text:
        combined = combined.drop_duplicates(subset=["text"]).copy()

    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(
        f"[fold aug/{aug_config}] Tá»-ng: {len(combined)} máº«u "
        f"(train={len(train_df)} + aug={len(combined) - len(train_df)})"
    )

    return combined


# â”€â”€ Utility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_class_weights(labels, num_classes=3):
    """TÃ­nh inverse-frequency class weights cho imbalanced datasets."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)


def verify_no_data_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Kiá»ƒm tra overlap exact text giá»¯a train vÃ  test.
    """
    train_texts = set(train_df["text"].astype(str).tolist())
    test_texts = set(test_df["text"].astype(str).tolist())
    overlap = train_texts & test_texts

    if overlap:
        print(f"[WARNING] Data leakage: {len(overlap)} máº«u xuáº¥t hiá»‡n cáº£ train láº«n test!")
        return False

    print(f"[OK] KhÃ´ng cÃ³ data leakage (train={len(train_texts)}, test={len(test_texts)} máº«u).")
    return True
