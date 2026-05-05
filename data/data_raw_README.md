# Data Directory Guide

This repository expects the dataset files to be placed manually in `data/raw/`.

## Expected files

```text
data/raw/train.csv
data/raw/val.csv
data/raw/test.csv
```

## Expected schema

Each CSV file should contain at least the following columns:

- `text`: Vietnamese review / sentence
- `label`: sentiment label

## Label mapping

- `0`: negative
- `1`: neutral
- `2`: positive

## Data access note

If the original dataset is subject to license, copyright, or redistribution restrictions,
do not upload the raw dataset files to the public repository.

Instead:

1. keep only this `README.md` file inside `data/raw/`,
2. describe the official dataset source in the main repository `README.md`,
3. and instruct users to obtain the raw files from the original provider.

## Augmented data

Generated augmentation outputs may be placed in:

```text
data/augmented/
```

Typical files include:

- `train_sr.csv`
- `train_bt.csv`

These files may also be excluded from Git if they are large or derived from restricted raw data.
