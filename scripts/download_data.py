"""
Download & prepare the fake-news dataset for this project.
=========================================================

Source: Kaggle  -> https://www.kaggle.com/datasets/saratchendra/fake-news
That dataset ships `train.csv` with the columns:

    id, title, author, text, label        (label: 1 = fake, 0 = real)

This script downloads it, builds the single combined `text` field the notebook
expects, normalises the label to the project convention (1 = fake, 0 = real),
and writes a clean CSV to `data/raw/fake_news.csv` — exactly where the notebook's
`DATA_PATH` already points.

Usage
-----
    # from the project root, inside the conda env:
    python scripts/download_data.py

    # only the article body as the text field:
    python scripts/download_data.py --text-fields text

    # if you already downloaded train.csv manually, skip the Kaggle call:
    python scripts/download_data.py --from-csv path/to/train.csv

Kaggle credentials
------------------
The download needs a Kaggle API token. One-time setup:
  1. Kaggle -> Account -> "Create New API Token"  (downloads kaggle.json)
  2. Put it at  ~/.kaggle/kaggle.json   (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)
If you can't set that up, download train.csv from the dataset page in a browser
and run this script with  --from-csv  (no credentials needed).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import pandas as pd

DATASET = "saratchendra/fake-news"

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(HERE, ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_PATH = os.path.join(RAW_DIR, "fake_news.csv")


def _find_train_csv(folder: str) -> str:
    """Locate train.csv (or the only/largest CSV) inside a downloaded folder."""
    csvs = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".csv"):
                csvs.append(os.path.join(root, f))
    if not csvs:
        raise FileNotFoundError(f"No CSV file found under {folder!r}")
    # Prefer a file literally called train.csv; otherwise the largest CSV.
    for c in csvs:
        if os.path.basename(c).lower() == "train.csv":
            return c
    return max(csvs, key=os.path.getsize)


def _download_via_kagglehub() -> str | None:
    try:
        import kagglehub
    except ImportError:
        return None
    print(f"Downloading {DATASET!r} via kagglehub …")
    path = kagglehub.dataset_download(DATASET)
    print(f"  cached at: {path}")
    return _find_train_csv(path)


def _download_via_cli() -> str | None:
    """Fallback: use the `kaggle` CLI if it's installed."""
    if shutil.which("kaggle") is None:
        return None
    import subprocess
    import tempfile

    tmp = tempfile.mkdtemp(prefix="fake_news_")
    print(f"Downloading {DATASET!r} via the kaggle CLI …")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", DATASET, "-p", tmp, "--unzip"],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  kaggle CLI failed: {exc}")
        return None
    return _find_train_csv(tmp)


def get_train_csv(from_csv: str | None) -> str:
    if from_csv:
        if not os.path.exists(from_csv):
            sys.exit(f"--from-csv path does not exist: {from_csv}")
        return from_csv

    path = _download_via_kagglehub() or _download_via_cli()
    if path is None:
        sys.exit(
            "Could not download the dataset automatically.\n"
            "  • Install a downloader:  pip install kagglehub   (or the `kaggle` CLI)\n"
            "  • And set up credentials:  ~/.kaggle/kaggle.json\n"
            "  • Or download train.csv from the dataset page and rerun with:\n"
            "        python scripts/download_data.py --from-csv path/to/train.csv\n"
            f"  Dataset: https://www.kaggle.com/datasets/{DATASET}"
        )
    return path


def prepare(train_csv: str, text_fields: list[str]) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    print(f"\nRead {len(df):,} rows from {train_csv}")
    print(f"  columns: {list(df.columns)}")

    missing = [c for c in text_fields if c not in df.columns]
    if missing:
        sys.exit(f"Requested text field(s) not in the data: {missing}")
    if "label" not in df.columns:
        sys.exit("Expected a 'label' column (1 = fake, 0 = real) and didn't find one.")

    # Build the combined text field, treating missing pieces as empty strings.
    text = df[text_fields[0]].fillna("").astype(str)
    for col in text_fields[1:]:
        text = text.str.cat(df[col].fillna("").astype(str), sep=" ")
    text = text.str.strip()

    out = pd.DataFrame({"text": text, "label": df["label"]})
    # Kaggle convention already matches ours (1 = fake, 0 = real); coerce + clean.
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    before = len(out)
    out = out[(out["text"].str.len() > 0) & out["label"].isin([0, 1])]
    out["label"] = out["label"].astype(int)
    out = out.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"  kept {len(out):,} rows after dropping empty/dup/invalid ({before - len(out):,} removed)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--text-fields",
        nargs="+",
        default=["title", "author", "text"],
        help="Column(s) from train.csv to concatenate into the model's text input "
             "(default: title author text).",
    )
    ap.add_argument(
        "--from-csv",
        default=None,
        help="Use a train.csv you already have instead of downloading from Kaggle.",
    )
    args = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    train_csv = get_train_csv(args.from_csv)
    out = prepare(train_csv, args.text_fields)
    out.to_csv(OUT_PATH, index=False)

    counts = out["label"].value_counts().rename({0: "real (0)", 1: "fake (1)"})
    print(f"\nWrote {len(out):,} rows -> {OUT_PATH}")
    print("Class balance:")
    print(counts.to_string())
    print("\nDone. Now run the notebook (notebooks/fake_news_detection.ipynb) top to bottom — "
          "it will pick this file up automatically via DATA_PATH.")


if __name__ == "__main__":
    main()
