from pathlib import Path
import pandas as pd

OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

FILES = {
    "Original Training": Path("data/raw/train.csv"),
    "+ Synonym Replacement (SR)": Path("data/augmented/train_sr.csv"),
    "+ Back-Translation (BT)": Path("data/augmented/train_bt.csv"),
    "Validation": Path("data/raw/val.csv"),
    "Test": Path("data/raw/test.csv"),
}

def read_csv(path):
    if not path.exists():
        return None
    return pd.read_csv(path)

def count_labels(df):
    counts = df["label"].value_counts().to_dict()
    return {
        "Negative": int(counts.get(0, 0)),
        "Neutral": int(counts.get(1, 0)),
        "Positive": int(counts.get(2, 0)),
    }

train_df = read_csv(FILES["Original Training"])
if train_df is None:
    raise FileNotFoundError("Missing data/raw/train.csv")

train_total = len(train_df)

rows = []

for name, path in FILES.items():
    df = read_csv(path)

    if df is None:
        rows.append({
            "Data Split": name,
            "Total": "MISSING",
            "Negative": "",
            "Neutral": "",
            "Positive": "",
            "Increase Rate": "",
            "File": str(path),
        })
        continue

    label_counts = count_labels(df)

    if name in ["Original Training", "Validation", "Test"]:
        increase_rate = "-"
    else:
        increase_rate = f"{((len(df) / train_total) * 100):.1f}% of original"
    
    rows.append({
        "Data Split": name,
        "Total": len(df),
        "Negative": label_counts["Negative"],
        "Neutral": label_counts["Neutral"],
        "Positive": label_counts["Positive"],
        "Increase Rate": increase_rate,
        "File": str(path),
    })

# Combined SR + BT + original train
sr_df = read_csv(FILES["+ Synonym Replacement (SR)"])
bt_df = read_csv(FILES["+ Back-Translation (BT)"])

if sr_df is not None and bt_df is not None:
    combined_df = pd.concat([train_df, sr_df, bt_df], ignore_index=True)
    label_counts = count_labels(combined_df)

    rows.insert(3, {
        "Data Split": "SR + BT Combined",
        "Total": len(combined_df),
        "Negative": label_counts["Negative"],
        "Neutral": label_counts["Neutral"],
        "Positive": label_counts["Positive"],
        "Increase Rate": f"+{((len(combined_df) - train_total) / train_total * 100):.1f}%",
        "File": "train + train_sr + train_bt",
    })

out = pd.DataFrame(rows)
out_path = OUT_DIR / "table1_dataset_statistics.csv"
out.to_csv(out_path, index=False, encoding="utf-8-sig")

print("[OK] Saved:", out_path)
print(out.to_string(index=False))
# '@ | Set-Content -Path ".\scripts_generate_table1.py" -Encoding UTF8

# python .\scripts_generate_table1.py