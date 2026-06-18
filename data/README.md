# Data

The actual datasets are **not** committed to the repo (see `.gitignore`). This file explains
what to put here and how the notebook reads it.

## Where files go

```
data/
├── raw/         <- put your original, untouched dataset here (e.g. fake_news.csv)
└── processed/   <- optional: any intermediate/cleaned files you generate
```

## Expected format

The notebook expects a **CSV with at least two columns**:

| column | meaning |
|---|---|
| a text column  | the article body or headline (free text) |
| a label column | whether the article is fake or real |

The notebook **auto-detects** these columns by common names:

- text column candidates: `text`, `content`, `article`, `body`, `news`, `headline`, `title`
- label column candidates: `label`, `target`, `class`, `y`, `is_fake`, `fake`, `category`

If your columns have different names, set `TEXT_COLUMN` / `LABEL_COLUMN` explicitly in the
notebook's **Section 2 (configuration)**.

## Label convention

Internally the project uses **`1 = fake`, `0 = real`** (positive class = fake). The loader
normalises many common encodings automatically:

- numeric `0` / `1`
- strings like `fake` / `real`, `false` / `true`, `unreliable` / `reliable`

If your labels use some other encoding, set `FAKE_LABEL_VALUE` in Section 2 to the value that
means "fake" (e.g. `FAKE_LABEL_VALUE = "FAKE"`).

## No dataset yet?

That's fine — if `DATA_PATH` does not point to an existing file, the notebook automatically
builds a small **synthetic, balanced** fake-vs-real dataset so every cell still runs end to end.
Swapping in the real data is a one-line change to `DATA_PATH`.

## Suggested public datasets

If you still need data, common choices for this task include the Kaggle "Fake and Real News"
dataset (separate `Fake.csv` / `True.csv` files — concatenate them and add a label column),
the WELFake dataset, or the LIAR dataset.
