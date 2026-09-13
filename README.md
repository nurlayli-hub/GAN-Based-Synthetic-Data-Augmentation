# GAN-Based Synthetic Data Augmentation for Multi-Class Diabetes Classification

Code for the Jurnal RESTI 2026 manuscript. Nine notebooks: preprocessing, four
tabular GAN architectures, synthetic-data quality analysis, baseline model
screening, classification, and a comparison notebook that settles whether
augmentation helped.

This folder is the corrected pipeline. `github_en - Copy` is the previous
revision, kept untouched for reference.

**Start with [PIPELINE.md](PIPELINE.md)** — the run order, the GPU requirements,
and a table of what every notebook reads and writes.

```bash
pip install -r requirements.txt
```

---

## What was wrong, and what was done about it

Five defects were found in the previous revision. Each is described below with
the fix.

### 1. Three of the four generators produced impossible values

CWGAN, TableGAN and MedGAN end their generator in a `Sigmoid`:

```python
nn.Linear(256, input_dim),
nn.Sigmoid()
```

No rounding, clipping or categorical handling follows, so every one of the 36
columns receives a continuous value in `[0, 1]`. But 24 of those columns can
only hold a small set of levels:

| Group | Count | Legal values after Min-Max scaling |
|---|---|---|
| Ordinal (Physical Activity, Socioeconomic Factors, Alcohol Consumption) | 3 | `{0.0, 0.5, 1.0}` |
| Binary (Dietary Habits … Early Onset Symptoms) | 16 | `{0.0, 1.0}` |
| One-hot (`Urine Test_*`) | 4 | exactly one `1` per row |
| **Neurological Assessments** | 1 | `{0.0, 0.5, 1.0}` |
| Continuous (Age, BMI, Insulin Levels, …) | 12 | any value in range |

A synthetic patient could therefore have `Smoking Status = 0.63`.

`Neurological Assessments` is the column whose KDE plot was visibly wrong. It is
numeric in the raw file, so it never appeared in the encoder mapping table and
notebook 02 listed it under `continuous_columns` — but it only ever takes three
levels.

| Notebook | Discrete-aware | Post-processing |
|---|---|---|
| 02 CTGAN | yes (`discrete_columns`) | not needed |
| 03 CWGAN | no | no |
| 04 TableGAN | no | no |
| 05 MedGAN | no | no |

**Fix.** Two cells inserted into each augmentation notebook. The first derives,
from `df_train` itself, which columns are discrete and which levels each may
take — nothing is hard-coded. The second runs between generation and saving: it
snaps each discrete column to its nearest legal level, reduces each one-hot group
to a single `1` by arg-max, clips continuous columns to the observed range, and
prints a before/after validity report.

CTGAN gets a validity **check** rather than a fix. If its report shows 100%
while the other three show near zero, that isolates native discrete handling as
the reason CTGAN behaves differently — a result worth reporting.

### 2. Preprocessing leaked test information

The old order was: clean all 70,000 records, then split. The Q1/Q3 thresholds
were therefore computed partly from rows that later became the test set, and the
test set was itself stripped of outliers.

**Fix.** Split first, learn the bounds from the training pool only, then apply
those same bounds to the test pool.

| Step | Cell | What it does |
|---|---|---|
| 1 | 16 | Partition the raw data, fit IQR bounds on the **training pool only**, apply to both |
| 2 | 20 | Zero handling, per partition (parameter-free, cannot leak) |
| 3 | 21 | Build the balanced test set, plus a leakage audit and a per-class headroom table |
| 4 | 24 | Feature / target separation |
| 5 | 26 | Encoding fitted on training only; mappings kept in `ENCODERS` for replay |
| 6 | 29 | Min-Max scaling fitted on training only |
| 7 | 31 | Export |

The test set is drawn from a raw candidate pool of 1,400 per class, leaving
headroom for the rows cleaning removes, then trimmed to exactly 1,000. Unused
candidates return to training rather than being discarded.

**The record counts change.** 63,401 / 50,401 no longer hold. Only the 13,000
balanced test set is unchanged by design. Tables 2 and 4 of the manuscript must
be refilled from the new output.

### 3. A third partition for realistic conditions

After the leakage fix `df_test` is clean, but it is still *filtered* — outlier
rows and zero-holding rows were removed. Real records arrive without that filter.

**Fix.** `UNSEEN_PER_CLASS = 100` in notebook 01 holds out 1,300 records from
the **raw** data before any cleaning, exported as `df_unseen.csv`. Notebook 09
scores it with the same fitted models and reports the gap.

Taking rows out of `df_train` after cleaning would have produced a set exactly as
tidy as `df_test` and measured nothing new — which is why the partition has to be
carved before cleaning.

### 4. The learning curves never converged

The search space capped `n_estimators` at 200 while the `learning_rate` grid
still offered 0.01 — a combination that guarantees the model never converges,
which is what the rising curves showed. Three further faults: `early_stopping(10)`
never fired, all three classifiers wrote to the same
`learning_curve_mean_5folds.png`, and every fold was truncated to the shortest
history.

**Fix.** Ceiling of 2,000 rounds with patience 50, the final model taking the
median `best_iteration` across folds; NaN padding instead of truncation;
a separate file per classifier; and an automatic convergence verdict printed
with each curve.

### 5. No baseline was ever measured

The manuscript claims augmentation improves performance, but the results table
contained only augmented configurations. There was no un-augmented row, so the
claim had never been tested.

**Fix.** Notebook 09. See below.

---

## Notebook 09 — the comparison

Every training set against one fixed test set, with every classifier.

| Scenario | Training set | Question it answers |
|---|---|---|
| 1 | `df_train.csv` | What does the data give with no augmentation? |
| 2 | `<stem>.csv` | Does augmentation add anything? |
| 3 | `<stem>_synthetic_only.csv` | Did the generator learn the real structure at all? |

Scenario 3 is the sharpest test of synthetic quality available. If synthetic data
alone cannot train a usable classifier, the generator has not captured the real
structure, and scenario 2 works only because the original rows inside it carry
the load.

It reports:

1. Accuracy, macro F1 and macro ROC-AUC for every combination.
2. **McNemar's test** against scenario 1. With 13,000 test records the standard
   error of accuracy is about 0.26 percentage points, so a raw gap of a few
   tenths means nothing on its own.
3. **Per-class recall**, from the scarcest class upward — augmentation may lift
   the rare classes while costing the common ones.
4. **Curated versus realistic**: the same models scored on `df_unseen.csv`.
5. An accuracy cross-check against the confusion matrix, and a separation audit
   of every training set against the test set.

The hyperparameter search runs once per classifier, on scenario 1, so a
difference between scenarios reflects the training data rather than the tuning.

### Reading the outcome

| Outcome | What to write |
|---|---|
| Scenario 2 significantly better | Augmentation helps; report where and by how much |
| No significant difference | Even with valid synthetic values, augmentation brings no benefit in this regime |
| Scenario 2 significantly worse | Synthetic data is harmful here; the mode collapse in the KDE plots is the explanation |

---

## Answering the reviewer

**Result discrepancies.** Every metrics cell recomputes accuracy as
`trace(cm) / sum(cm)`, prints it beside `accuracy_score`, and asserts the two
agree. A gap can then only mean the figure and the table came from different
runs. A per-class table prints support, correct and errors straight from the
matrix, so a 1000/1000 claim is traceable.

**Leakage and the perfectly classified classes.** Three layers of evidence, all
printed automatically: the separation audit reports how many feature vectors the
training and test sets share (it should be zero); the test partition is held out
before augmentation, so no generator ever saw it; and notebook 09 reproduces the
same perfect result from the **un-augmented baseline**, which no generator can
be responsible for.

**Reproducibility.** Notebooks 08 and 09 print every seed in one block — global,
cross-validation, search, estimators, and the generation seed used in 02–05 —
ready to paste into the manuscript.

---

## Standardised output

All four augmentation notebooks write exactly two files under one naming rule:

```
<model>_<device>_epoch<E>_augment<T>_seed<S>.csv                 original + synthetic
<model>_<device>_epoch<E>_augment<T>_seed<S>_synthetic_only.csv  synthetic rows only
```

The redundant and hard-coded save cells were removed. Notebook 02 previously
wrote `ctgan_perclass_epoch100_augment6000.csv` with the epoch baked into the
string, so the 300- and 500-epoch runs silently overwrote each other. Its
`train_ctgan_per_class` now returns the synthetic rows separately instead of only
the merged frame.

Before writing, each export asserts that the combined frame really is original +
synthetic, that the column order matches, and that neither frame holds NaN.

---

## Runtime

`RandomizedSearchCV` costs 25 fits and was previously repeated for every training
set and every classifier — about three hours of the total runtime, for values
that barely move between training sets.

```
first run   TUNE = True    searches, writes best_params.json
later runs  TUNE = False   reads it back, no search
```

Two further switches: `RECORD_CURVE = False` skips the training-side learning
curve for runs whose figure does not go into the paper (results are unchanged),
and `USE_GPU = True` moves XGBoost onto the GPU. `learning_rate = 0.01` was
dropped from the search space, since 0.05 and 0.1 converge in a few hundred
rounds instead of a few thousand.

Together these cut the full 13-configuration sweep from roughly 17 hours to
about 6.
