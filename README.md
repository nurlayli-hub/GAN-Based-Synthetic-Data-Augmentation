# GAN-Based Synthetic Data Augmentation for Multi-Class Diabetes Classification

Code for the Jurnal RESTI 2026 manuscript. Eight notebooks — preprocessing, four
tabular GAN architectures, synthetic-data quality analysis, baseline model
screening, and classification — plus a class-weighted variant of the
classification notebook that provides the baseline every augmented run is
measured against.

The notebooks carry only the comments needed to follow the code. Every banner,
rationale and design note that used to sit inside them lives in the three files
beside them, so the reasoning is still on record without crowding the cells: the
run order and the per-notebook reference in `PIPELINE.md`, the design decisions
and the answers to the reviewer in this file, and the environment in
`requirements.txt`. No executable line differs from the annotated version.

**Start with [PIPELINE.md](PIPELINE.md)** — the run order, the GPU requirements,
and a table of what every notebook reads and writes. Its closing section,
*Notebook notes*, holds the two long explanatory cells that were moved out of
notebooks 01 and 08.

**[GAN_architecture_table.docx](GAN_architecture_table.docx)** documents the four
generators exactly as this code implements them, including where each departs
from the published architecture it is named after. Three of the four are custom
PyTorch implementations rather than reference implementations, so the manuscript
has to describe them from this table rather than by citing the original papers
alone.

```bash
pip install -r requirements.txt
```

---

## Contents

| File | Role |
|---|---|
| `01_data_preprocessing_v2.ipynb` | Partitioning, leakage-free cleaning, encoding, scaling |
| `02_augmentation_ctgan_v2.ipynb` | CTGAN, one model per class, with Drive checkpointing |
| `03_augmentation_cwgan_v2.ipynb` | Conditional WGAN-GP |
| `04_augmentation_tablegan_v2.ipynb` | Unconditional MLP GAN + rejection sampling |
| `05_augmentation_medgan_v2.ipynb` | Autoencoder + latent-space GAN + rejection sampling |
| `06_synthetic_data_quality_v2.ipynb` | KDE overlays, KS test, Wasserstein distance |
| `07_baseline_model_screening_v2.ipynb` | 21 classifiers screened on `df_train` |
| `08_classification_and_evaluation_v2.ipynb` | The evaluation notebook, one run per training set |
| `08_… - class weighting (for unbalanced data).ipynb` | The same notebook with inverse-frequency class weights |
| `GAN_architecture_table.docx` | Architecture and training configuration of the four generators |

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

No rounding, clipping or categorical handling followed, so every column received
a continuous value in `[0, 1]`. But most of those columns can only hold a small
set of levels:

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
notebook 02 lists it under `continuous_columns` — but it only ever takes three
levels.

| Notebook | Discrete-aware | Post-processing before the fix |
|---|---|---|
| 02 CTGAN | partly — 23 columns + target declared, `Neurological Assessments` not among them | not applied |
| 03 CWGAN | no | none |
| 04 TableGAN | no | none |
| 05 MedGAN | no | none |

**Fix.** Two cells inserted into each augmentation notebook. The first derives,
from `df_train` itself, which columns are discrete and which levels each may
take — nothing is hard-coded, so the rule stays correct if the preprocessing
changes. The second runs between generation and saving: it snaps each discrete
column to its nearest legal level, reduces each one-hot group to a single `1` by
arg-max, clips continuous columns to the observed range, and prints a
before/after validity report.

CTGAN gets the same cell as a **check**. If its report shows near 100% on the
declared columns while the other three show near zero, that isolates native
discrete handling as the reason CTGAN behaves differently — a result worth
reporting. `Neurological Assessments` is the one column the check can still
change for CTGAN, because it was never declared.

### 2. Preprocessing leaked test information

The old order was: clean all 70,000 records, then split. The Q1/Q3 thresholds
were therefore computed partly from rows that later became the test set, and the
test set was itself stripped of outliers.

**Fix.** Split first, learn the bounds from the training pool only, then apply
those same bounds to the test pool.

| Step | Cell | What it does |
|---|---|---|
| 1 | 16 | Hold out the unseen partition from the raw data, partition the rest, fit IQR bounds on the **training pool only**, apply to both |
| 2 | 20 | Zero handling, per partition (parameter-free, cannot leak) |
| 3 | 21 | Build the balanced test set, plus a leakage audit and a per-class headroom table |
| 4 | 24 | Feature / target separation |
| 5 | 26 | Encoding fitted on training only; mappings kept in `ENCODERS` for replay |
| 6 | 29 | Min-Max scaling fitted on training only |
| 7 | 31 | Export |

The test set is drawn from a raw candidate pool of 1,400 per class, leaving
headroom for the rows cleaning removes, then trimmed to exactly 1,000. Unused
candidates return to training rather than being discarded.

**The record counts change.** 63,401 / 50,401 no longer hold; `df_train` is
about 48.4k rows. Only the 13,000 balanced test set is unchanged by design. Any
manuscript table reporting record counts must be refilled from the notebook 01
output.

### 3. A third partition for realistic conditions

After the leakage fix `df_test` is clean, but it is still *filtered* — outlier
rows and zero-holding rows were removed. Real records arrive without that filter.

**Fix.** `UNSEEN_PER_CLASS = 100` in notebook 01 holds out 1,300 records from
the **raw** data before any cleaning, exported as `df_unseen.csv`. Notebook 08
scores it by pointing its test-file line at `df_unseen.csv` instead of
`df_test.csv`; nothing else changes.

Taking rows out of `df_train` after cleaning would have produced a set exactly as
tidy as `df_test` and measured nothing new — which is why the partition has to be
carved before cleaning.

At 1,300 records the 95% margin of error on accuracy is about ±1.6 percentage
points, wider than the spread between configurations. Use it for one headline
figure, never for ranking.

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

**Fix.** Two baselines, both produced by notebook 08.

| Baseline | How to run it |
|---|---|
| Plain | `08_classification_and_evaluation_v2.ipynb` with `TRAINING_FILE = 'df_train.csv'` |
| Class-weighted | `08_… - class weighting (for unbalanced data).ipynb` on the same file |

The class-weighted variant weights every class by inverse frequency inside the
hyperparameter search, inside each fold and in the final fit. LightGBM and
Random Forest take this as `class_weight`; XGBoost, which has no such parameter
for multi-class problems, receives the equivalent per-row sample weights. Fold
weights are recomputed from the fold's own distribution, so nothing crosses a
fold boundary.

It costs one run and no synthetic data, and it is the number every augmented
configuration has to beat.

---

## Answering the reviewer

**Result discrepancies.** Every metrics cell recomputes accuracy as
`trace(cm) / sum(cm)`, prints it beside `accuracy_score`, and reports the
difference. A gap can then only mean the figure and the table came from
different runs. A per-class table prints support, correct and errors straight
from the matrix, so a 1000/1000 claim is traceable.

**Leakage and the perfectly classified classes.** Three layers of evidence, all
printed automatically: the separation audit reports how many feature vectors the
training and test sets share (it should be zero); the test partition is held out
before augmentation, so no generator ever saw it; and the un-augmented baseline
reproduces the same perfect classes, which no generator can be responsible for.

**Reproducibility.** Notebook 08 prints every seed in one block — global,
cross-validation, search, estimators, and the generation seed used in 02–05 —
ready to paste into the manuscript.

**What the four models actually are.** Only CTGAN comes from a published
package (`ctgan 0.12.1`). CWGAN, TableGAN and MedGAN are custom PyTorch
implementations that share the name of a published architecture without
reproducing it: the TableGAN here has no convolutional encoder, auxiliary
classifier or information loss, and the MedGAN here has no minibatch averaging
or shortcut connections. `GAN_architecture_table.docx` sets out every difference
side by side. The manuscript should describe them from that table.

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

## Two things to record when reporting these runs

**"Epoch" is not one unit across the four notebooks.** Notebook 02 counts passes
over a class's rows in mini-batches of 64. Notebook 03 counts generator updates,
each preceded by five critic updates on a random mini-batch of 256 — 500 epochs
is about 16 effective passes over the data. Notebooks 04 and 05 count
full-batch updates. The 100 / 300 / 500 settings are comparable within an
architecture, not between architectures.

**TableGAN and MedGAN may emit perturbed real rows.** When rejection sampling
cannot reach a class quota within the attempt limit (50 for TableGAN, 30 for
MedGAN), the shortfall is filled by bootstrap resampling of that class's real
records with Gaussian noise σ = 0.005. Those rows land in the `_synthetic_only`
file and therefore in the KS and Wasserstein figures. Each notebook prints how
many were added per class; that number bounds how much of the "synthetic" data
is genuinely generated, and it belongs in the paper.

---

## Runtime

`RandomizedSearchCV` costs 25 fits and was previously repeated for every training
set and every classifier — about three hours of the total runtime, for values
that barely move between training sets.

```
first run   TUNE = True    searches, writes best_params.json
later runs  TUNE = False   reads it back, no search
```

`best_params.json` holds one entry per classifier name, so all three share one
file without colliding.

Two further switches: `RECORD_CURVE = False` skips the training-side learning
curve for runs whose figure does not go into the paper (results are unchanged),
and `USE_GPU = True` moves XGBoost onto the GPU. `learning_rate = 0.01` was
dropped from the search space, since 0.05 and 0.1 converge in a few hundred
rounds instead of a few thousand.

Together these cut the full sweep from roughly 17 hours to about 6.
