# deep_learning_fake_news

Fake-news detection with **RNNs and LSTMs** — a Deep Learning final-project MVP for IE University.

The project takes the raw text of a news article (or headline) and predicts whether it is **fake**
or **real**. It pairs an analytical backend (a Jupyter notebook) with a Streamlit frontend.

---

## 1. Business case

Moderation teams and platforms receive far more articles per day than humans can fact-check.
Misinformation that slips through erodes trust; over-blocking real reporting damages credibility.
This model is a **triage assistant**: it scores incoming articles so reviewers see a ranked queue
instead of an undifferentiated feed.

**Value proposition:** saves reviewer time, lets a small team cover far more content, and reduces the
trust/brand-safety cost of misinformation reaching readers.

**The two errors are not equal** (this drives the metrics we report):

| Error | Meaning | Cost |
|---|---|---|
| **False negative** | fake article predicted real | misinformation reaches readers — the costlier error |
| **False positive** | real article predicted fake | wasted reviewer time, risk of over-censoring |

Because false negatives usually cost more, we report **recall (on fake)** and **ROC-AUC** alongside accuracy.

---

## 2. Repository structure

```
deep_learning_fake_news/
├── README.md                     <- project overview
├── environment.yml               <- conda environment
├── .gitignore
├── data/
│   ├── README.md                 <- dataset format + how the loader auto-detects columns
│   ├── raw/                      <- put the original dataset here (git-ignored)
│   └── processed/                <- optional intermediate files (git-ignored)
├── notebooks/
│   └── fake_news_detection.ipynb <- model: EDA → preprocessing → 3 models → eval → export
├── models/                       <- exported model lands here (git-ignored, regenerated)
├── app/
│   └── app.py                    <- Streamlit frontend
└── reports/
    └── figures/                  <- plots saved by the notebook
```

---

## 3. Quickstart

### a. Create the environment

```bash
conda env create -f environment.yml
conda activate deep_learning_fake_news
```

### b. Get the dataset

Download Kaggle's [`saratchendra/fake-news`](https://www.kaggle.com/datasets/saratchendra/fake-news)
`train.csv` (columns `id, title, author, text, label`; `label` is `1 = fake` / `0 = real`) and save it as
`data/raw/fake_train.csv` — exactly where the notebook's `DATA_PATH` points.

All cleaning is done **inside the notebook** (Section 3): it selects the `text` and `label` columns, drops
empty rows and duplicates, and normalises the label. If you skip this step entirely, the notebook still
runs on a built-in synthetic dataset.

### c. Run the notebook

```bash
jupyter lab     # or: jupyter notebook
```

Open `notebooks/fake_news_detection.ipynb` and run all cells. With no dataset present it trains on a
built-in synthetic dataset so everything works immediately. The final section exports
`models/fake_news_model.keras`.

### d. Launch the frontend

From the **project root** (after the notebook has exported the model):

```bash
streamlit run app/app.py
```

Paste a full article, optionally adjust the decision threshold in the sidebar, and the app
returns a verdict, a confidence score, and the raw fake-probability.

---

## 4. Using the real dataset

The default path is `data/raw/fake_train.csv` (Kaggle's `train.csv`, see Quickstart §b) — the notebook is
already configured to read it (`DATA_PATH`, `TEXT_COLUMN = "text"`) and cleans it in Section 3.

To use a **different** CSV instead:

1. Drop your CSV into `data/raw/` and point `DATA_PATH` at it (notebook **Section 2**).
2. If your columns differ, set `TEXT_COLUMN`, `LABEL_COLUMN`, and/or `FAKE_LABEL_VALUE` there.
3. Re-run the notebook top to bottom.

The loader normalises labels to `1 = fake`, `0 = real` and handles common encodings
(`0/1`, `fake/real`, `true/false`, …). See `data/README.md` for the full format spec.

---

## 5. The model

Text preprocessing:

- a custom **`standardization`** function: lowercase → strip HTML → strip punctuation → drop non-ASCII;
- a **`TextVectorization`** layer (`max_tokens=10000`, `output_sequence_length=250`), **adapted on the
  training split only** to avoid leakage;
- a learned **`Embedding`** (dim 64) front-end.

Three recurrent architectures are trained and compared:

| Model | Architecture | Role |
|---|---|---|
| **SimpleRNN** | `Embedding → SimpleRNN(32) → sigmoid` | baseline / lower bound |
| **Stacked LSTM** | `Embedding → LSTM(64, return_sequences) → LSTM(32) → Dense → Dropout → sigmoid` | gated memory for long-range dependencies |
| **Bidirectional LSTM** | `Embedding → BiLSTM(64) → BiLSTM(32) → Dense → Dropout → sigmoid` | reads text both directions; usually strongest |

All use **`BinaryCrossentropy`** loss and **`Adam(1e-4)`**, with **`EarlyStopping`** (overfitting control)
and **`ModelCheckpoint`** (keeps best-validation weights). Data is split **stratified 70/15/15**
(train/val/test) and the test set is only used for the final evaluation.

The notebook exports a single **end-to-end model** that bundles the vectorizer inside it, so the app
feeds in raw text and reads out a probability — no preprocessing code is duplicated in the frontend.

---

## 6. Results

Test-set performance on the Kaggle `saratchendra/fake-news` dataset (positive class = fake):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SimpleRNN | 0.793 | 0.827 | 0.730 | 0.775 | 0.816 |
| Stacked LSTM | 0.802 | 0.908 | 0.663 | 0.767 | 0.873 |
| Bidirectional LSTM | **0.953** | **0.958** | **0.945** | **0.951** | **0.989** |

The Bidirectional LSTM is the strongest model and the one exported for the app. Confusion matrices,
ROC curves, and training curves are saved to `reports/figures/`.

---

## 7. Conventions

- **Label convention:** `1 = fake`, `0 = real`, everywhere.
- **No leakage:** the vectorizer is `adapt()`-ed on training text only; the test set is touched only at
  final evaluation.
- **Single notebook:** all modelling lives in `notebooks/fake_news_detection.ipynb`. The only other code
  is the frontend `app/app.py`.
- **Config in one place:** every knob (`DATA_PATH`, vocab size, sequence length, epochs, …) is in the
  notebook's Section 2.
- **Reproducibility:** seeds are fixed (`SEED = 42`).
- **Smoke-test switch:** setting the env var `FAKE_NEWS_SMOKE_TEST=1` shrinks the data and epochs for a
  fast end-to-end check; leave it unset for a full run.
