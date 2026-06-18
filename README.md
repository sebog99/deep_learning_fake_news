# deep_learning_fake_news

Fake-news detection with **RNNs and LSTMs** — a Deep Learning final-project MVP for IE University.

The project takes the raw text of a news article (or headline) and predicts whether it is **fake**
or **real**. It pairs an analytical backend (a single Jupyter notebook) with a user-facing frontend
(a Streamlit app), demonstrating an end-to-end, real-time predictive system.

> **This README is the source of truth.** If you are picking the project up to continue it, read this
> file first — it describes the goal, the structure, the conventions, and exactly what is left to do.

---

## 1. The business case (the "why")

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
├── README.md                     <- you are here (source of truth)
├── environment.yml               <- conda environment
├── .gitignore
├── data/
│   ├── README.md                 <- dataset format + how the loader auto-detects columns
│   ├── raw/                      <- put the original dataset here (git-ignored)
│   └── processed/                <- optional intermediate files (git-ignored)
├── notebooks/
│   └── fake_news_detection.ipynb <- THE model: EDA → preprocessing → 3 models → eval → export
├── models/                       <- exported model lands here (git-ignored, regenerated)
├── app/
│   └── app.py                    <- Streamlit frontend (the MVP)
└── reports/
    └── figures/                  <- plots saved by the notebook for the slide deck
```

---

## 3. Quickstart

### a. Create the environment

```bash
conda env create -f environment.yml
conda activate deep_learning_fake_news
```

### b. Run the notebook

```bash
jupyter lab     # or: jupyter notebook
```

Open `notebooks/fake_news_detection.ipynb` and run all cells. With no dataset present it trains on a
built-in synthetic dataset so everything works immediately. The final section exports
`models/fake_news_model.keras`.

### c. Launch the frontend (MVP)

From the **project root** (after the notebook has exported the model):

```bash
streamlit run app/app.py
```

Paste a headline or article, optionally adjust the decision threshold in the sidebar, and the app
returns a verdict, a confidence score, and the raw fake-probability.

---

## 4. Using the real dataset

1. Drop your CSV into `data/raw/` (e.g. `data/raw/fake_news.csv`).
2. In the notebook's **Section 2 (configuration)**, set `DATA_PATH` to point at it.
3. If the auto-detection guesses the wrong columns, also set `TEXT_COLUMN`, `LABEL_COLUMN`, and/or
   `FAKE_LABEL_VALUE`.
4. Re-run the notebook top to bottom.

The loader normalises labels to `1 = fake`, `0 = real` and handles common encodings
(`0/1`, `fake/real`, `true/false`, …). See `data/README.md` for the full format spec.

---

## 5. The model (the "how")

Text preprocessing mirrors the class sentiment-analysis pipeline:

- a custom **`standardization`** function: lowercase → strip HTML → strip punctuation;
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

## 6. How the project maps to the grading rubric

| Rubric pillar (weight) | Where it is addressed |
|---|---|
| Business use case & value proposition (20%) | README §1 and notebook Section 1 (incl. false-positive/negative cost analysis) |
| Technical depth & model architecture (25%) | notebook Sections 2–9: leakage-free preprocessing, 3 justified architectures, hyperparameters, early stopping |
| MVP integration & frontend UX (25%) | `app/app.py` + the end-to-end model export (notebook Section 11) |
| Presentation & team delivery (20%) | figures in `reports/figures/` + the architecture/results story below |
| Live demo & time management (10%) | `streamlit run app/app.py` for the live demo |

---

## 7. Results

Fill this in after running on the real dataset (the notebook prints this exact table in Section 9):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SimpleRNN | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Stacked LSTM | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Bidirectional LSTM | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

Confusion matrices, ROC curves, and training curves are saved to `reports/figures/`.

---

## 8. Conventions (read before continuing the project)

- **Label convention:** `1 = fake`, `0 = real`, everywhere.
- **No leakage:** the vectorizer is `adapt()`-ed on training text only; the test set is touched only at
  final evaluation.
- **Single notebook:** all modelling lives in `notebooks/fake_news_detection.ipynb` (course requirement).
  The only other code is the frontend `app/app.py`.
- **Config in one place:** every knob (`DATA_PATH`, vocab size, sequence length, epochs, …) is in the
  notebook's Section 2.
- **Reproducibility:** seeds are fixed (`SEED = 42`).
- **Smoke-test switch:** setting the env var `FAKE_NEWS_SMOKE_TEST=1` shrinks the data and epochs for a
  fast end-to-end check; leave it unset for real runs.

---

## 9. Roadmap

- [ ] Integrate the real labelled course dataset (replace the synthetic fallback via `DATA_PATH`).
- [ ] Re-run the notebook and fill in the results table (§7) and figures.
- [ ] Rehearse the 15-minute presentation + live Streamlit demo.
- [ ] (Optional) tune sequence length / embedding dim / threshold for the precision–recall trade-off you want.

---

## 10. Deliverables (per the guidelines)

1. **GitHub repository** — this repo: documented backend (notebook) + frontend (`app.py`) + this README.
2. **Presentation deck** (PDF) — built separately; use the figures in `reports/figures/`.
