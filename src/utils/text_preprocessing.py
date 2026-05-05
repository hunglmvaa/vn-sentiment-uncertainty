"""
src/utils/text_preprocessing.py
---------------------------------
Vietnamese word segmentation before feeding into PhoBERT.

PhoBERT was trained on word-segmented Vietnamese (e.g. "sản_phẩm", "giao_hàng").
Passing raw text (un-segmented) degrades its representations significantly.

This module provides a single entry-point: segment_text(text) -> str

Backend priority:
  1. underthesea  (easy install, good accuracy, recommended for most setups)
  2. raw text     (fallback if neither is installed — prints ONE warning)

Install:
    pip install underthesea

Usage:
    from src.utils.text_preprocessing import segment_text

    segmented = segment_text("sản phẩm này rất tốt nhưng giao hàng chậm")
    # → "sản_phẩm này rất tốt nhưng giao_hàng chậm"
"""

import re
import warnings
from functools import lru_cache

# ── Try underthesea ────────────────────────────────────────────────────────────
try:
    from underthesea import word_tokenize as _uts_tokenize
    _HAS_UNDERTHESEA = True
except ImportError:
    _HAS_UNDERTHESEA = False

_WARNED = False   # print the fallback warning only once


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces; strip leading/trailing."""
    return re.sub(r"\s+", " ", text).strip()


def segment_text(text: str) -> str:
    """
    Segment Vietnamese text for PhoBERT input.

    Returns a string where multi-syllable words are joined by underscores,
    consistent with vinai/phobert-base's tokenizer expectations.

    Example:
        "sản phẩm tốt nhưng giao hàng chậm"
      → "sản_phẩm tốt nhưng giao_hàng chậm"

    Safe to call with None or non-string input — returns "" in that case.
    """
    global _WARNED

    if not isinstance(text, str) or not text.strip():
        return ""

    text = _normalize_whitespace(text)

    if _HAS_UNDERTHESEA:
        try:
            result = _uts_tokenize(text)

            if isinstance(result, str):
                return _normalize_whitespace(result)

            if isinstance(result, (list, tuple)):
                return _normalize_whitespace(" ".join(map(str, result)))

        except Exception:
            pass

    # ── Fallback: raw text ────────────────────────────────────────────────
    if not _WARNED:
        warnings.warn(
            "[text_preprocessing] underthesea not found. "
            "Passing raw (un-segmented) text to PhoBERT — results will be "
            "sub-optimal. Install with: pip install underthesea",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True
    return text


def segment_dataframe(df, text_col: str = "text", out_col: str = "text") -> "pd.DataFrame":
    """
    Apply segment_text to an entire DataFrame column in-place (or to a copy).

    Parameters
    ----------
    df       : pd.DataFrame with at least `text_col`
    text_col : source column name (default "text")
    out_col  : destination column name — set same as text_col to overwrite

    Returns the modified DataFrame (copy if out_col != text_col).
    """
    import pandas as pd

    df = df.copy()
    df[out_col] = df[text_col].apply(segment_text)
    return df
