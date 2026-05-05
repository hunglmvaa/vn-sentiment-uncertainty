# Vietnamese Sentiment Analysis with Uncertainty-Aware Augmentation

Source code and experiment package for the study:

**Improving Vietnamese Sentiment Analysis Using Data Augmentation and Transformer-Based Models: An Uncertainty-Aware Approach**

## 1. Purpose of this repository

This repository is intended to serve as:

- the source-code archive for the experiments reported in the paper,
- a reproducibility package for reviewers and readers,
- and a technical evidence package containing scripts, configurations, logs, and result tables.

The repository focuses on Vietnamese sentiment classification with:

- traditional baselines,
- PhoBERT-based supervised learning,
- Synonym Replacement (SR),
- Back Translation (BT),
- MC-Dropout uncertainty estimation,
- and UGA (Uncertainty-Guided Augmentation).

---

## 2. Repository structure

```text
fixed_v6_package/
├── data/
│   ├── raw/                      # Input CSV files: train.csv, val.csv, test.csv
│   └── augmented/                # Generated SR/BT training files
├── experiments/
│   ├── run_baselines.py
│   ├── run_phobert.py
│   ├── run_phobert_3seeds_final_aug.py
│   ├── run_uncertainty_analysis.py
│   ├── run_uga.py
│   └── run_uga_full.py
├── src/
│   ├── augmentation/
│   │   ├── synonym_replacement.py
│   │   └── back_translation.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── uncertainty.py
│   │   └── bootstrap_test.py
│   ├── models/
│   │   ├── phobert_classifier.py
│   │   └── baseline_models.py
│   └── utils/
│       ├── data_loader.py
│       └── logger.py
├── results/                      # Output JSON / CSV / checkpoints / tables
├── logs/                         # Execution logs
├── requirements.txt
└── README.md
```

---

## 3. Dataset

### 3.1 Expected local format

Place the dataset files in:

```text
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

Expected columns:

- `text`: Vietnamese review / sentence
- `label`: sentiment label

Label mapping:

- `0`: negative
- `1`: neutral
- `2`: positive

### 3.2 Data availability note

If the original dataset cannot be redistributed publicly, this repository should not upload the raw data directly.

Instead:

- keep only the expected folder structure,
- provide a short `data/README.md`,
- describe the dataset source, license, and access procedure,
- and explain how to place the files into `data/raw/`.

This is the recommended practice for publication-oriented repositories.

---

## 4. Environment setup

Recommended Python version:

- Python 3.10 or 3.11

Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate       # Windows PowerShell

pip install --upgrade pip
pip install -r requirements.txt
```

If using Kaggle or Colab, install directly with:

```bash
pip install -r requirements.txt
```

---

## 5. Main experiment workflow

### 5.1 Synonym Replacement (SR)

```bash
python src/augmentation/synonym_replacement.py     --input data/raw/train.csv     --output data/augmented/train_sr.csv     --rate 0.15     --seed 42
```

### 5.2 Back Translation (BT)

```bash
python src/augmentation/back_translation.py     --input data/raw/train.csv     --output data/augmented/train_bt.csv     --sim_threshold 0.75     --batch_size 16
```

### 5.3 PhoBERT experiments

Example full run:

```bash
python experiments/run_phobert_3seeds_final_aug.py     --data_dir data     --config all     --seeds 42 123 2024     --n_folds 5     --mc_passes 20     --epochs 10     --batch_size 32     --patience 3     --use_final_aug_files     --sr_file data/augmented/train_sr.csv     --bt_file data/augmented/train_bt.csv
```

### 5.4 Uncertainty analysis

```bash
python experiments/run_uncertainty_analysis.py --results_dir results
```

### 5.5 UGA debug run

```bash
python experiments/run_uga.py     --data_dir data     --max_rounds 1     --mc_passes 20     --ensemble_seeds 42     --alpha 0.80     --beta 0.40     --viwordnet ./viwordnet_clean.tsv     --bt_batch_size 16     --force
```

### 5.6 Full UGA experiment

Use `run_uga_full.py` when the goal is to execute the full multi-fold protocol.

---

## 6. Reproducibility notes

### 6.1 Recommended evidence to keep in the repository

For publication support, keep:

- source code in `src/` and `experiments/`,
- `requirements.txt`,
- important result tables in `results/*.csv`,
- selected logs in `logs/`,
- and a clear description of the protocol in this README.

### 6.2 Recommended evidence to keep outside GitHub

Avoid pushing large or sensitive files directly to GitHub:

- raw dataset files if redistribution is restricted,
- large checkpoints,
- Hugging Face cache,
- temporary files,
- notebook outputs that are not essential.

For large checkpoints, prefer:

- Google Drive,
- Hugging Face Hub,
- Kaggle Dataset,
- or Zenodo archive.

### 6.3 Canonical experiment settings

Unless otherwise stated, the repository uses:

- PhoBERT as the main transformer model,
- SR and BT as augmentation methods,
- entropy-based uncertainty partitioning for UGA,
- and fixed seeds for controlled comparison.

If a paper submission uses a frozen code snapshot, create a Git tag such as:

```bash
git tag -a v1.0-paper-submission -m "Code used for paper submission"
git push origin v1.0-paper-submission
```

---

## 7. Suggested `.gitignore` policy

Recommended exclusions:

- virtual environments,
- Python cache,
- temporary logs,
- Hugging Face cache,
- large checkpoints,
- raw private data.

A typical `.gitignore` should include at least:

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
env/
venv/
results/checkpoints/
*.pt
*.bin
*.ckpt
.DS_Store
Thumbs.db
```

---

## 8. How to cite or refer to this repository in a paper

Suggested wording:

The source code, experiment scripts, and reproducibility package are available in the project repository. A frozen version used for submission is identified by the release tag `v1.0-paper-submission`.

If long-term archival is required, it is recommended to archive the tagged release on Zenodo and cite the DOI.

---

## 9. Contact and maintenance

This repository is maintained as an experiment package for research verification and reproducibility.
For reproducibility questions, include:

- script name,
- command used,
- hardware environment,
- and log excerpt.

This makes debugging and verification substantially easier.
