
"""
augmentation/synonym_replacement.py
------------------------------------
Robust Synonym Replacement (SR) augmentation for Vietnamese text.

Key improvements:
  1. Reject pseudo-augmentation caused only by tokenization / spacing changes.
  2. Preserve source mapping with source_id and source_text for fold-safe loading.
  3. Load ViWordNet robustly from JSON, simple TSV/CSV, and ViWordNet-like TSV dumps.
  4. Replace only true content tokens with explicit lexical change tracking.
  5. Filter suspicious outputs and cap aggressive replacement depth.

Recommended usage with local ViWordNet TSV:
  python src/augmentation/synonym_replacement.py ^
      --input data/raw/train.csv ^
      --output data/augmented/train_sr.csv ^
      --rate 0.15 ^
      --seed 42 ^
      --min_changes 1 ^
      --max_changes 2 ^
      --viwordnet src/augmentation/viwordnet.tsv
"""

import os
import re
import csv
import json
import random
import argparse
from typing import List, Optional, Dict, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import vncorenlp
    HAS_VNCORENLP = True
except ImportError:
    HAS_VNCORENLP = False

try:
    from underthesea import pos_tag as uts_pos_tag
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


STOP_WORDS = {
    "và", "của", "là", "có", "được", "cho", "trong", "với", "này",
    "đó", "các", "những", "một", "như", "thì", "đã", "đang", "sẽ",
    "không", "rất", "cũng", "bởi", "vì", "nên", "mà", "nhưng", "hoặc",
    "hay", "tuy", "dù", "nếu", "khi", "bao", "nhiêu", "hơn", "nhất",
    "cả", "mọi", "từ", "đến", "ra", "vào", "lên", "xuống", "đây",
    "kia", "ấy", "tôi", "bạn", "họ", "chúng", "mình", "anh", "chị",
    "em", "ông", "bà", "nó", "hắn",
}

SENTIMENT_TOKENS = {
    "tốt", "hay", "đẹp", "xuất_sắc", "tuyệt", "vời", "thích", "yêu",
    "hài_lòng", "tuyệt_vời", "hoàn_hảo", "chất_lượng", "nhanh", "tiện",
    "tệ", "xấu", "kém", "chán", "ghét", "thất_vọng", "tồi", "bực",
    "khó_chịu", "cẩu_thả", "chậm", "đắt", "hỏng", "lỗi", "lừa",
    "rất", "cực", "siêu", "quá", "lắm", "vô_cùng",
}

SYNONYM_DICT = {
    "sản_phẩm": ["mặt_hàng", "vật_phẩm"],
    "giá": ["mức_giá", "chi_phí"],
    "shop": ["cửa_hàng", "gian_hàng"],
    "người_bán": ["người_bán_hàng", "chủ_shop"],
    "giao_hàng": ["vận_chuyển", "giao_nhận"],
    "đơn_hàng": ["đơn_đặt", "đơn_mua"],
    "màu_sắc": ["tông_màu", "gam_màu"],
    "kích_thước": ["kích_cỡ", "cỡ"],
    "bao_bì": ["đóng_gói"],
    "khách_hàng": ["người_mua", "khách"],
    "thời_gian": ["khoảng_thời_gian", "thời_điểm"],
    "hình_ảnh": ["hình", "ảnh"],
    "mô_tả": ["miêu_tả", "thông_tin"],
    "mua": ["đặt_hàng", "chọn_mua"],
    "nhận": ["tiếp_nhận", "lấy"],
    "gửi": ["chuyển", "vận_chuyển"],
    "đặt": ["chọn", "đăng_ký"],
    "dùng": ["sử_dụng", "xài"],
    "thấy": ["nhận_thấy", "cảm_thấy"],
    "muốn": ["mong_muốn", "mong"],
    "nhanh": ["mau", "kịp_thời", "đúng_hạn"],
    "chậm": ["trễ", "muộn", "lâu"],
    "đẹp": ["xinh", "bắt_mắt", "hấp_dẫn"],
    "xấu": ["kém_đẹp", "thô"],
    "to": ["lớn", "cỡ_lớn"],
    "nhỏ": ["bé", "nhỏ_gọn"],
    "mới": ["hiện_đại", "cập_nhật"],
    "cũ": ["cũ_kỹ", "đã_qua_sử_dụng"],
    "rẻ": ["phải_chăng", "hợp_lý"],
    "đắt": ["giá_cao", "tốn_kém"],
    "ổn": ["ổn_định", "chấp_nhận_được"],
}

SAFE_TOKEN_PATTERN = re.compile(r"^[\wÀ-ỹ_]+$", re.UNICODE)
PUNCT_SPACING_RE = re.compile(r"\s+([,.;:!?%])")
MULTISPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[\wÀ-ỹ_]+", re.UNICODE)
POS_LIKE = {
    "adj", "adv", "noun", "verb", "adjective", "adverb",
    "noun_act", "noun.act", "noun_attribute", "noun.attribute",
    "noun_body", "noun.body", "noun_cognition", "noun.cognition",
    "noun_communication", "noun.communication", "noun_event", "noun.event",
    "noun_feeling", "noun.feeling", "noun_food", "noun.food",
    "noun_group", "noun.group", "noun_location", "noun.location",
    "noun_motive", "noun.motive", "noun_object", "noun.object",
    "noun_person", "noun.person", "noun_phenomenon", "noun.phenomenon",
    "noun_plant", "noun.plant", "noun_possession", "noun.possession",
    "noun_process", "noun.process", "noun_quantity", "noun.quantity",
    "noun_relation", "noun.relation", "noun_shape", "noun.shape",
    "noun_state", "noun.state", "noun_substance", "noun.substance",
    "noun_time", "noun.time", "verb_body", "verb.body",
    "verb_change", "verb.change", "verb_cognition", "verb.cognition",
    "verb_communication", "verb.communication", "verb_competition", "verb.competition",
    "verb_consumption", "verb.consumption", "verb_contact", "verb.contact",
    "verb_creation", "verb.creation", "verb_emotion", "verb.emotion",
    "verb_motion", "verb.motion", "verb_perception", "verb.perception",
    "verb_possession", "verb.possession", "verb_social", "verb.social",
    "verb_stative", "verb.stative", "verb_weather", "verb.weather",
}

def default_viwordnet_path() -> str:
    return os.path.join(os.path.dirname(__file__), "viwordnet.tsv")


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = MULTISPACE_RE.sub(" ", text)
    text = PUNCT_SPACING_RE.sub(r"\1", text)
    return text.strip()


def normalize_token(token: str) -> str:
    token = str(token).strip().lower().replace(" ", "_")
    token = re.sub(r"_+", "_", token)
    return token


def lexical_tokens(text: str) -> List[str]:
    return [normalize_token(t) for t in WORD_RE.findall(str(text)) if t.strip()]


def is_surface_only_change(src: str, aug: str) -> bool:
    return normalize_text(src) == normalize_text(aug)


def changed_content_tokens(src: str, aug: str) -> int:
    src_tokens = lexical_tokens(src)
    aug_tokens = lexical_tokens(aug)
    changed = sum(1 for a, b in zip(src_tokens, aug_tokens) if a != b)
    changed += abs(len(src_tokens) - len(aug_tokens))
    return changed


def parse_synonym_record(key: str, values: List[str], out: Dict[str, List[str]]) -> None:
    norm_key = normalize_token(key)
    if not norm_key or norm_key in POS_LIKE:
        return

    filtered = []
    for v in values:
        norm_v = normalize_token(v)
        if not norm_v:
            continue
        if norm_v == norm_key:
            continue
        if norm_v in POS_LIKE:
            continue
        if norm_v in SENTIMENT_TOKENS:
            continue
        if not SAFE_TOKEN_PATTERN.match(norm_v):
            continue
        if any(ch.isdigit() for ch in norm_v):
            continue
        filtered.append(norm_v)

    if filtered:
        out.setdefault(norm_key, [])
        out[norm_key].extend(filtered)


def _maybe_parse_simple_tsv_row(row: List[str], loaded: Dict[str, List[str]]) -> bool:
    if len(row) != 2:
        return False
    key = normalize_token(row[0])
    values = re.split(r"[,;|]", row[1])
    parse_synonym_record(key, values, loaded)
    return True


def _maybe_parse_viwordnet_dump_row(row: List[str], loaded: Dict[str, List[str]]) -> bool:
    # Typical patterns:
    #   adj    bằng    ngang bằng    tương xứng ...
    #   noun.act    học_tập    sự học ...
    if len(row) < 3:
        return False

    first = normalize_token(row[0])
    if first not in POS_LIKE:
        return False

    lemma = row[1]
    synonyms = row[2:]
    parse_synonym_record(lemma, synonyms, loaded)
    return True


def load_viwordnet(path: Optional[str] = None) -> Dict[str, List[str]]:
    merged: Dict[str, List[str]] = {k: list(v) for k, v in SYNONYM_DICT.items()}

    if not path:
        auto_path = default_viwordnet_path()
        if os.path.exists(auto_path):
            path = auto_path

    if not path or not os.path.exists(path):
        print("[ViWordNet] File not found — using built-in synonym dict.")
        for k, vals in merged.items():
            merged[k] = sorted(set(vals))
        return merged

    loaded: Dict[str, List[str]] = {}
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, values in data.items():
                    if isinstance(values, list):
                        parse_synonym_record(key, values, loaded)
                    elif isinstance(values, str):
                        parse_synonym_record(key, re.split(r"[,;|]", values), loaded)
        else:
            # Prefer deterministic parsing over csv.Sniffer for ViWordNet-like resources.
            # *.tsv  -> split by tab
            # *.csv  -> split by comma
            delimiter = "\t" if ext == ".tsv" else ","

            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                for line_num, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:
                        continue

                    if delimiter == "\t":
                        row = [c.strip() for c in line.split("\t") if str(c).strip()]
                    else:
                        row = [c.strip() for c in line.split(",") if str(c).strip()]

                    if len(row) < 2:
                        continue

                    # Skip common header rows
                    if line_num == 1 and normalize_token(row[0]) in {"category", "pos", "type", "word", "lemma"}:
                        continue

                    if _maybe_parse_viwordnet_dump_row(row, loaded):
                        continue
                    _maybe_parse_simple_tsv_row(row, loaded)
    except Exception as exc:
        print(f"[ViWordNet] Failed to parse {path}: {exc}")

    for key, values in loaded.items():
        merged.setdefault(key, [])
        merged[key].extend(values)

    for key, values in merged.items():
        merged[key] = sorted(set(v for v in values if v and v != key))

    print(f"[ViWordNet] Loaded {len(loaded)} external entries from {path}; total={len(merged)}")
    return merged


class SynonymReplacer:
    """Applies conservative synonym replacement augmentation to Vietnamese text."""

    def __init__(
        self,
        rate: float = 0.15,
        synonym_dict: Optional[Dict[str, List[str]]] = None,
        vncorenlp_dir: Optional[str] = None,
        seed: int = 42,
        min_changes: int = 1,
        max_changes: int = 2,
        max_change_ratio: float = 0.20,
    ):
        self.rate = rate
        self.syns = synonym_dict or load_viwordnet()
        self.rng = random.Random(seed)
        self.min_changes = max(1, int(min_changes))
        self.max_changes = max(self.min_changes, int(max_changes))
        self.max_change_ratio = max(0.05, float(max_change_ratio))
        self.nlp = None
        self._init_tagger(vncorenlp_dir)

    def _init_tagger(self, vncorenlp_dir: Optional[str]) -> None:
        if HAS_VNCORENLP and vncorenlp_dir:
            try:
                self.nlp = vncorenlp.VnCoreNLP(
                    os.path.join(vncorenlp_dir, "VnCoreNLP-1.1.1.jar"),
                    annotators="wseg,pos",
                    max_heap_size="-Xmx512m",
                )
                self.tagger = "vncorenlp"
                print("[Tagger] Using VnCoreNLP")
                return
            except Exception as exc:
                print(f"[Tagger] VnCoreNLP failed: {exc}")

        if HAS_UNDERTHESEA:
            self.tagger = "underthesea"
            print("[Tagger] Using underthesea (fallback)")
        else:
            self.tagger = "none"
            print("[Tagger] No tagger available — conservative whitespace fallback")

    def _pos_tag(self, text: str) -> List[Tuple[str, str]]:
        if self.tagger == "vncorenlp" and self.nlp:
            try:
                ann = self.nlp.annotate(text)
                result = []
                for sent in ann.get("sentences", []):
                    for tok in sent:
                        result.append((tok["form"], tok["posTag"]))
                return result
            except Exception:
                pass

        if self.tagger == "underthesea":
            try:
                return [(w, t) for w, t in uts_pos_tag(text)]
            except Exception:
                pass

        return [(w, "N") for w in str(text).split()]

    def _is_eligible(self, word: str, pos: str) -> bool:
        w = normalize_token(word)
        if not w:
            return False
        if w in STOP_WORDS or w in SENTIMENT_TOKENS:
            return False
        if w not in self.syns:
            return False
        if any(ch.isdigit() for ch in w):
            return False
        if len(w) <= 1:
            return False

        content_pos = {
            "N", "V", "A", "R", "Np", "Nc", "Nu", "Ny",
            "Vb", "Va", "Adj", "Adv", "NOUN", "VERB", "ADJ", "ADV"
        }
        pos_ok = pos in content_pos or pos[:1].upper() in {"N", "V", "A", "R"}
        return pos_ok

    def _choose_synonym(self, token_key: str) -> Optional[str]:
        candidates = self.syns.get(token_key, [])
        if not candidates:
            return None

        safe_candidates = [
            c for c in candidates
            if c != token_key
            and c not in SENTIMENT_TOKENS
            and c not in POS_LIKE
            and SAFE_TOKEN_PATTERN.match(c)
            and len(c) > 1
            and not any(ch.isdigit() for ch in c)
        ]
        if not safe_candidates:
            return None
        return self.rng.choice(safe_candidates)

    def augment(self, text: str) -> Tuple[str, int]:
        tokens = self._pos_tag(str(text))
        eligible_idx = [i for i, (w, p) in enumerate(tokens) if self._is_eligible(w, p)]
        if not eligible_idx:
            return str(text), 0

        planned_changes = max(1, int(round(len(eligible_idx) * self.rate)))
        planned_changes = min(planned_changes, self.max_changes)
        replace_idx = set(self.rng.sample(eligible_idx, min(planned_changes, len(eligible_idx))))

        result_tokens: List[str] = []
        actual_changes = 0

        for idx, (word, _) in enumerate(tokens):
            if idx in replace_idx:
                key = normalize_token(word)
                repl = self._choose_synonym(key)
                if repl is not None and repl != key:
                    result_tokens.append(repl)
                    actual_changes += 1
                    continue
            result_tokens.append(word)

        aug_text = " ".join(result_tokens).strip()
        return aug_text, actual_changes

    def _is_valid_output(self, src_text: str, aug_text: str, n_changes: int) -> bool:
        if not aug_text.strip():
            return False
        if n_changes < self.min_changes or n_changes > self.max_changes:
            return False
        if is_surface_only_change(src_text, aug_text):
            return False
        content_changes = changed_content_tokens(src_text, aug_text)
        if content_changes < self.min_changes:
            return False
        src_len = max(1, len(lexical_tokens(src_text)))
        if content_changes / src_len > self.max_change_ratio:
            return False
        if normalize_text(src_text) == normalize_text(aug_text):
            return False
        return True

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("Input DataFrame must contain ['text', 'label'] columns.")

        work_df = df.copy().reset_index(drop=True)
        work_df["text"] = work_df["text"].astype(str)
        work_df["label"] = work_df["label"].astype(int)

        if "source_id" not in work_df.columns:
            work_df["source_id"] = work_df.index.astype(str)
        else:
            work_df["source_id"] = work_df["source_id"].astype(str)

        rows = []
        seen_aug_texts = set()

        for _, row in tqdm(work_df.iterrows(), total=len(work_df), desc="Synonym replacement"):
            src_text = str(row["text"]).strip()
            aug_text, n_changes = self.augment(src_text)
            aug_norm = normalize_text(aug_text)

            if not self._is_valid_output(src_text, aug_text, n_changes):
                continue
            if aug_norm in seen_aug_texts:
                continue

            seen_aug_texts.add(aug_norm)
            rows.append(
                {
                    "source_id": str(row["source_id"]),
                    "source_text": src_text,
                    "text": aug_text,
                    "label": int(row["label"]),
                    "sr_changes": int(n_changes),
                }
            )

        return pd.DataFrame(rows, columns=["source_id", "source_text", "text", "label", "sr_changes"])

    def close(self) -> None:
        if self.nlp is not None:
            try:
                self.nlp.close()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Synonym Replacement Augmentation")
    parser.add_argument("--input", required=True, help="Input CSV containing text, label")
    parser.add_argument("--output", required=True, help="Output augmented CSV")
    parser.add_argument("--rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vncorenlp_dir", default=None, help="Path to VnCoreNLP dir")
    parser.add_argument("--viwordnet", default=default_viwordnet_path(), help="Path to ViWordNet or synonym resource")
    parser.add_argument("--min_changes", type=int, default=1, help="Minimum lexical changes required")
    parser.add_argument("--max_changes", type=int, default=2, help="Maximum lexical changes allowed")
    parser.add_argument("--max_change_ratio", type=float, default=0.20, help="Maximum changed-token ratio allowed")
    args = parser.parse_args()

    print(
        f"[SR] Rate={args.rate}, seed={args.seed}, min_changes={args.min_changes}, "
        f"max_changes={args.max_changes}, max_change_ratio={args.max_change_ratio}"
    )
    print(f"[SR] ViWordNet path={args.viwordnet}")

    syn_dict = load_viwordnet(args.viwordnet)
    replacer = SynonymReplacer(
        rate=args.rate,
        synonym_dict=syn_dict,
        vncorenlp_dir=args.vncorenlp_dir,
        seed=args.seed,
        min_changes=args.min_changes,
        max_changes=args.max_changes,
        max_change_ratio=args.max_change_ratio,
    )

    df = pd.read_csv(args.input)
    print(f"[SR] Input: {len(df)} samples")

    aug_df = replacer.augment_dataframe(df)
    print(f"[SR] Augmented: {len(aug_df)} new samples")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    aug_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[SR] Saved to {args.output}")

    replacer.close()


if __name__ == "__main__":
    main()
