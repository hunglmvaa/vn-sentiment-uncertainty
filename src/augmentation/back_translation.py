"""
augmentation/back_translation.py
--------------------------------
Back-Translation (BT) augmentation for Vietnamese text.

Key improvements over the previous version:
  1. Stronger semantic filtering with PhoBERT cosine similarity.
  2. Reject pseudo-valid outputs with low lexical overlap or excessive shortening.
  3. Preserve source mapping and quality metadata for auditability.
  4. Filter noisy translations containing abnormal tokens or broken punctuation.
  5. Conservative defaults: higher similarity threshold and stricter safety checks.

Usage:
  python src/augmentation/back_translation.py \
      --input data/raw/train.csv \
      --output data/augmented/train_bt.csv \
      --sim_threshold 0.75 \
      --batch_size 16
"""

import os
import re
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModel


VI2EN_MODEL = "Helsinki-NLP/opus-mt-vi-en"
EN2VI_MODEL = "Helsinki-NLP/opus-mt-en-vi"
PHOBERT_MODEL = "vinai/phobert-base"

MULTISPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[\wÀ-ỹ_]+", re.UNICODE)
ABNORMAL_TOKEN_RE = re.compile(r"[{}\[\]<>|`~^]+")
REPEATED_PUNCT_RE = re.compile(r"([!?.,:;])\1{2,}")
BROKEN_WORD_RE = re.compile(r"\b(?:[A-Za-z]{1,3}\s){3,}[A-Za-z]{1,3}\b")

TECHNICAL_PROTECTED_TERMS = {
    "android", "ios", "samsung", "iphone", "ipad", "ssd", "hdd", "ram",
    "cpu", "gpu", "main", "flash", "bb", "passport", "phobert", "mbert"
}


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = MULTISPACE_RE.sub(" ", text)
    return text


def lexical_tokens(text: str) -> List[str]:
    return [t.lower() for t in WORD_RE.findall(normalize_text(text)) if t.strip()]


def lexical_overlap_ratio(src: str, aug: str) -> float:
    src_tokens = set(lexical_tokens(src))
    aug_tokens = set(lexical_tokens(aug))
    if not src_tokens:
        return 0.0
    return len(src_tokens & aug_tokens) / max(len(src_tokens), 1)


def length_ratio(src: str, aug: str) -> float:
    src_len = max(len(lexical_tokens(src)), 1)
    aug_len = len(lexical_tokens(aug))
    return aug_len / src_len


def has_suspicious_noise(text: str) -> bool:
    txt = normalize_text(text)
    if not txt:
        return True
    if ABNORMAL_TOKEN_RE.search(txt):
        return True
    if REPEATED_PUNCT_RE.search(txt):
        return True
    if BROKEN_WORD_RE.search(txt):
        return True
    return False


def protected_term_drift(src: str, aug: str) -> bool:
    src_tokens = set(lexical_tokens(src))
    aug_tokens = set(lexical_tokens(aug))
    protected_in_src = src_tokens & TECHNICAL_PROTECTED_TERMS
    if not protected_in_src:
        return False
    return not protected_in_src.issubset(aug_tokens)


class BackTranslator:
    """Vietnamese → English → Vietnamese back-translation with strong filtering."""

    def __init__(
        self,
        sim_threshold: float = 0.75,
        device: str = None,
        batch_size: int = 16,
        min_overlap: float = 0.35,
        max_overlap: float = 0.95,
        min_len_ratio: float = 0.60,
        max_len_ratio: float = 1.60,
    ):
        self.threshold = sim_threshold
        self.batch_size = batch_size
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.min_len_ratio = min_len_ratio
        self.max_len_ratio = max_len_ratio
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[BT] Device: {self.device}")
        self._load_models()

    def _load_models(self) -> None:
        print("[BT] Loading Vi→En translation model...")
        self.vi2en_tok = MarianTokenizer.from_pretrained(VI2EN_MODEL)
        self.vi2en_mdl = MarianMTModel.from_pretrained(VI2EN_MODEL).to(self.device)
        self.vi2en_mdl.eval()

        print("[BT] Loading En→Vi translation model...")
        self.en2vi_tok = MarianTokenizer.from_pretrained(EN2VI_MODEL)
        self.en2vi_mdl = MarianMTModel.from_pretrained(EN2VI_MODEL).to(self.device)
        self.en2vi_mdl.eval()

        print("[BT] Loading PhoBERT for similarity filtering...")
        self.phobert_tok = AutoTokenizer.from_pretrained(PHOBERT_MODEL)
        self.phobert_mdl = AutoModel.from_pretrained(PHOBERT_MODEL).to(self.device)
        self.phobert_mdl.eval()

    def _translate_batch(self, texts: List[str], tokenizer, model, max_length: int = 256) -> List[str]:
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        return tokenizer.batch_decode(output, skip_special_tokens=True)

    def translate_vi2en(self, texts: List[str]) -> List[str]:
        return self._translate_batch(texts, self.vi2en_tok, self.vi2en_mdl)

    def translate_en2vi(self, texts: List[str]) -> List[str]:
        return self._translate_batch(texts, self.en2vi_tok, self.en2vi_mdl)

    def _get_embeddings(self, texts: List[str]) -> torch.Tensor:
        try:
            from src.utils.text_preprocessing import segment_text
            segmented_texts = [segment_text(str(t)) for t in texts]
        except Exception:
            segmented_texts = [str(t) for t in texts]

        enc = self.phobert_tok(
            segmented_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            out = self.phobert_mdl(**enc)

        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(emb, dim=-1)

    def _cosine_similarities(self, orig_texts: List[str], bt_texts: List[str]) -> np.ndarray:
        orig_emb = self._get_embeddings(orig_texts)
        bt_emb = self._get_embeddings(bt_texts)
        return (orig_emb * bt_emb).sum(dim=-1).detach().cpu().numpy()

    def _validate_pair(self, src_text: str, bt_text: str, sim: float) -> Tuple[bool, dict]:
        src = normalize_text(src_text)
        bt = normalize_text(bt_text)

        meta = {
            "bt_sim": float(sim),
            "lex_overlap": float(lexical_overlap_ratio(src, bt)),
            "len_ratio": float(length_ratio(src, bt)),
        }

        if not bt or bt == src:
            return False, meta
        if sim < self.threshold:
            return False, meta
        if has_suspicious_noise(bt):
            return False, meta
        if protected_term_drift(src, bt):
            return False, meta
        if meta["lex_overlap"] < self.min_overlap:
            return False, meta
        if meta["lex_overlap"] >= self.max_overlap:
            return False, meta
        if meta["len_ratio"] < self.min_len_ratio or meta["len_ratio"] > self.max_len_ratio:
            return False, meta

        return True, meta

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply back-translation and keep only high-quality paraphrases."""
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Input DataFrame must contain ['text', 'label'] columns.")

        work_df = df.copy().reset_index(drop=True)
        work_df["text"] = work_df["text"].astype(str)
        work_df["label"] = work_df["label"].astype(int)

        if "source_id" not in work_df.columns:
            work_df["source_id"] = work_df.index.astype(str)
        else:
            work_df["source_id"] = work_df["source_id"].astype(str)

        texts = work_df["text"].tolist()
        labels = work_df["label"].tolist()
        source_ids = work_df["source_id"].tolist()

        accepted_rows = []
        seen_texts = set()
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(n_batches), desc="Back-translation"):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size

            batch_texts = texts[start:end]
            batch_labels = labels[start:end]
            batch_source_ids = source_ids[start:end]

            try:
                en_texts = self.translate_vi2en(batch_texts)
            except Exception as exc:
                print(f"[BT] Vi→En failed batch {i}: {exc}")
                continue

            try:
                back_vi_texts = self.translate_en2vi(en_texts)
            except Exception as exc:
                print(f"[BT] En→Vi failed batch {i}: {exc}")
                continue

            try:
                sims = self._cosine_similarities(batch_texts, back_vi_texts)
            except Exception as exc:
                print(f"[BT] Similarity computation failed batch {i}: {exc}")
                sims = np.zeros(len(batch_texts), dtype=np.float32)

            for orig_text, bt_text, lbl, sim, src_id in zip(
                batch_texts, back_vi_texts, batch_labels, sims, batch_source_ids
            ):
                ok, meta = self._validate_pair(orig_text, bt_text, float(sim))
                bt_norm = normalize_text(bt_text)
                if not ok:
                    continue
                if bt_norm in seen_texts:
                    continue

                seen_texts.add(bt_norm)
                accepted_rows.append(
                    {
                        "source_id": str(src_id),
                        "source_text": normalize_text(orig_text),
                        "text": bt_norm,
                        "label": int(lbl),
                        "bt_sim": meta["bt_sim"],
                        "lex_overlap": meta["lex_overlap"],
                        "len_ratio": meta["len_ratio"],
                    }
                )

        if accepted_rows:
            aug_df = pd.DataFrame(accepted_rows)
            avg_sim = aug_df["bt_sim"].mean()
        else:
            aug_df = pd.DataFrame(
                columns=[
                    "source_id", "source_text", "text", "label",
                    "bt_sim", "lex_overlap", "len_ratio"
                ]
            )
            avg_sim = float("nan")

        print(
            f"[BT] Accepted {len(aug_df)}/{len(texts)} samples "
            f"(threshold={self.threshold:.2f}, avg_sim={avg_sim:.4f})"
        )
        return aug_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-Translation Augmentation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sim_threshold", type=float, default=0.75)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min_overlap", type=float, default=0.35)
    parser.add_argument("--max_overlap", type=float, default=0.95)
    parser.add_argument("--min_len_ratio", type=float, default=0.60)
    parser.add_argument("--max_len_ratio", type=float, default=1.60)
    args = parser.parse_args()

    print(
        f"[BT] Threshold={args.sim_threshold}, batch_size={args.batch_size}, "
        f"min_overlap={args.min_overlap}, max_overlap={args.max_overlap}, "
        f"min_len_ratio={args.min_len_ratio}, max_len_ratio={args.max_len_ratio}"
    )

    bt = BackTranslator(
        sim_threshold=args.sim_threshold,
        device=args.device,
        batch_size=args.batch_size,
        min_overlap=args.min_overlap,
        max_overlap=args.max_overlap,
        min_len_ratio=args.min_len_ratio,
        max_len_ratio=args.max_len_ratio,
    )

    df = pd.read_csv(args.input)
    print(f"[BT] Input: {len(df)} samples")

    aug_df = bt.augment_dataframe(df)
    print(f"[BT] Augmented: {len(aug_df)} new samples")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    aug_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[BT] Saved to {args.output}")


if __name__ == "__main__":
    main()
