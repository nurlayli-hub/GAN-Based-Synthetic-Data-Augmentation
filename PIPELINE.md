# Pipeline Reference

Every notebook in this folder: what goes in, what it does, what comes out, and
whether it needs a GPU. Eight notebooks, one of them in two variants.

```
01_data_preprocessing_v2.ipynb
02_augmentation_ctgan_v2.ipynb
03_augmentation_cwgan_v2.ipynb
04_augmentation_tablegan_v2.ipynb
05_augmentation_medgan_v2.ipynb
06_synthetic_data_quality_v2.ipynb
07_baseline_model_screening_v2.ipynb
08_classification_and_evaluation_v2.ipynb
08_classification_and_evaluation_v2 - class weighting (for unbalanced data).ipynb
```

`GAN_architecture_table.docx` in this folder documents the four generative
architectures exactly as the code implements them, including where each one
departs from the published architecture it is named after.

---

## Run order

```
STAGE 1                 STAGE 2 (parallel)              STAGE 3              STAGE 4
┌──────────────┐    ┌──────────────────────────┐   ┌────────────────┐   ┌──────────┐
│      01      │    │  02  03  04  05          │   │       08       │   │    06    │
│ preprocessing│───▶│  CTGAN CWGAN Table Med   │──▶│ classification │──▶│ KDE + KS │
│              │    │                          │   │ + evaluation   │   │ + WD     │
│              │    │  07  model screening     │   │                │   │          │
└──────────────┘    └──────────────────────────┘   └────────────────┘   └──────────┘
   CPU                  GPU          CPU                GPU opt              CPU
```

**Stage 2 runs in two independent sessions.** Notebooks 02–05 and notebook 07
both read `df_train.csv` and produce outputs that never touch each other: 07
screens 21 baseline models to justify carrying LightGBM, XGBoost and Random
Forest forward, while 02–05 generate synthetic data. Run 07 on a **CPU runtime**
so it does not consume GPU quota.

**Stage 4 comes last on purpose.** Notebook 06 is run once, for the single
generative model that stage 3 identified as best, to produce the KDE figure and
the KS / Wasserstein table that go into the paper.

---

## GPU

| Notebook | GPU | Why |
|---|---|---|
| **01** preprocessing | **Not needed** | pandas and scikit-learn only |
| **02** CTGAN | **Required** | `enable_gpu=True`; the cell asserts the generator is on CUDA, and cell 2 raises if no GPU is present |
| **03** CWGAN | **Required** | PyTorch, WGAN-GP with 5 critic iterations per step; cell 2 raises if no GPU is present |
| **04** TableGAN | **Required** | PyTorch, plus rejection sampling in batches of 50,000 |
| **05** MedGAN | **Required** | PyTorch, autoencoder + latent GAN |
| **06** quality | **Not needed** | scipy and seaborn |
| **07** model screening | **Not needed** | 21 scikit-learn / boosting models, all CPU |
| **08** classification | **Optional** | Only XGBoost uses it (`USE_GPU`). LightGBM and Random Forest are CPU either way |

Practical consequence: **only stage 2's augmentation half genuinely needs the
GPU.** Notebook 07 can run at the same time on a separate CPU session. Stage 3
benefits from a GPU but will finish without one.

---

## Master table

| # | Notebook | Input | What it does | Output | GPU |
|---|---|---|---|---|---|
| **01** | `01_data_preprocessing_v2.ipynb` | Raw Kaggle CSV, 70,000 × 34 | EDA → duplicate and NaN check → **hold out the unseen partition from the raw data**, then partition train / test → fit IQR bounds on the training pool only and apply them to both → drop rows holding a zero, per partition → encode (3 ordinal, 16 binary, 1 one-hot group, LabelEncoder on the target) → Min-Max scale, fitted on training only. Prints a leakage audit and a per-class headroom table. | `df_train.csv`<br>`df_test.csv` (13,000, 1,000/class)<br>`df_unseen.csv` (1,300, raw) | No |
| **02** | `02_augmentation_ctgan_v2.ipynb` | `df_train.csv` | Builds the discrete-value schema from `df_train`, then trains **one CTGAN per class** (13 models), conditional, `discrete_columns` declared, `batch_size=64`, `pac=1`, up to 6,000 rows per class. Each class is checkpointed to Drive as it finishes, so a Colab disconnect costs only the class in progress. Asserts the generator is on CUDA. Runs the validity **check**. | `ctgan_perclass_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **03** | `03_augmentation_cwgan_v2.ipynb` | `df_train.csv` | Conditional WGAN-GP written in PyTorch. Label one-hot (13) concatenated to the noise and to the critic input, λ = 10, 5 critic iterations, mini-batch 256. Generates directly for the target class. Runs the validity **correction**. | `cwgan_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **04** | `04_augmentation_tablegan_v2.ipynb` | `df_train.csv` | Unconditional MLP GAN with BCE loss over the 49-column vector (36 features + 13 one-hot target), full batch. Class quotas are met by **rejection sampling** (50,000 per batch, up to 50 attempts); any shortfall is filled by bootstrap resampling with Gaussian noise σ = 0.005. Runs the validity correction. | `tablegan_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **05** | `05_augmentation_medgan_v2.ipynb` | `df_train.csv` | Autoencoder (49 → 256 → 64 → 256 → 49) plus a latent-space GAN, unconditional, full batch. Rejection sampling with the same fallback (up to 30 attempts). Runs the validity correction. | `medgan_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **07** | `07_baseline_model_screening_v2.ipynb` | `df_train.csv` | Screens **21 classifiers** on its own internal split of `df_train` (`test_size=0.2, random_state=123, stratify=y`) — it does **not** use `df_test`. Results are appended to a CSV on Drive, so an interrupted run resumes without re-fitting the models already done. Ranks by macro F1. | `hasil_evaluasi_model_fulldata.csv` on Drive; the ranking that justifies the three classifiers | No |
| **08** | `08_classification_and_evaluation_v2.ipynb` | one training set (`TRAINING_FILE`) + `df_test.csv` | Deep single-run notebook: separation audit, cached tuning, stratified 5-fold CV, converged learning curves, confusion matrix, per-class report, sensitivity / specificity, accuracy cross-check against the confusion matrix. | `learning_curve_lightgbm.png`<br>`learning_curve_xgboost.png`<br>`learning_curve_random_forest.png`<br>`best_params.json` | Optional |
| **08w** | `08_... - class weighting (for unbalanced data).ipynb` | `df_train.csv` + `df_test.csv` | The same notebook with one change: every class is weighted by inverse frequency inside the search, inside each fold and in the final fit. LightGBM and Random Forest take `class_weight`; XGBoost receives the equivalent per-row sample weights. Fold weights are recomputed from the fold's own distribution. | Same as 08, plus balanced accuracy | Optional |
| **06** | `06_synthetic_data_quality_v2.ipynb` | `df_train.csv` + **`<stem>_synthetic_only.csv`** of the winning model | Equal-size sampling per class, KDE overlays over 13 continuous variables, **Kolmogorov–Smirnov test** and **Wasserstein distance** per variable. Warns if the synthetic file still contains real records. | KDE figure and the KS / Wasserstein table | No |

---

## The three scenarios

Notebooks 02–05 each write **two** files, so three training scenarios can be
compared against one untouched test set:

| Scenario | Training set | File | Question it answers |
|---|---|---|---|
| 1 | Original only | `df_train.csv` | What does the data give with no augmentation? |
| 2 | Original + synthetic | `<stem>.csv` | Does augmentation add anything? |
| 3 | Synthetic only | `<stem>_synthetic_only.csv` | Did the generator learn the real structure at all? |

Notebook 08 runs whichever one `TRAINING_FILE` points at. Scenario 3 is the
sharpest test of synthetic quality available: if synthetic data alone cannot
train a usable classifier, the generator has not captured the real structure,
and scenario 2 works only because the original rows inside it carry the load.

---

## The three partitions

| Partition | Rows | Condition | Use |
|---|---|---|---|
| `df_train.csv` | ~48k | cleaned, imbalanced | training, and the input to augmentation |
| `df_test.csv` | 13,000 | cleaned, balanced 1,000/class | **ranks** the configurations |
| `df_unseen.csv` | 1,300 | **raw, never cleaned**, 100/class | one honest real-world number |

`df_unseen` is held out from the raw data **before** any cleaning, so it still
contains the outliers and zero-coded missing values that `df_test` had removed.

**How the unseen numbers are produced.** There is no separate notebook for it.
Notebook 08 reads its evaluation set in the INPUT SELECTION cell:

```python
df_test = pd.read_csv("df_test.csv")      # change to "df_unseen.csv"
```

Point that one line at `df_unseen.csv`, leave `TRAINING_FILE` and everything
else untouched, and re-run. The separation audit then reports 1,300 test rows
instead of 13,000, and every downstream cell scores the unseen partition. Keep
the two runs in separate copies of the notebook so the outputs are not
overwritten.

At 1,300 records the 95% margin of error on accuracy is about ±1.6 percentage
points — wider than the spread between configurations. **Use the unseen set for
one headline figure, never for ranking.** Ranking stays with `df_test` and its
13,000 records, where the margin is about ±0.5 points.

---

## File naming

```
<model>_<device>_epoch<E>_augment<T>_seed<S>.csv                 original + synthetic
<model>_<device>_epoch<E>_augment<T>_seed<S>_synthetic_only.csv  synthetic rows only
```

| Placeholder | Values |
|---|---|
| `<model>` | `ctgan_perclass`, `cwgan`, `tablegan`, `medgan` |
| `<device>` | `gpu`, `cpu` |
| `<E>` | `100`, `300`, `500` |
| `<T>` | `6000` |
| `<S>` | `42` |

Every filename carries the epoch count, so the three settings cannot overwrite
one another. Notebooks 06 and 08 rebuild the same string from `MODEL_TAG`,
`DEVICE_TAG`, `EPOCHS` and `TARGET_PER_CLASS`, so nothing is typed by hand.

---

## Switches

| Switch | Notebook | Meaning |
|---|---|---|
| `UNSEEN_PER_CLASS` | 01 | **100** — holds out 1,300 raw records before cleaning. 0 reproduces the two-partition design |
| `TEST_PER_CLASS` | 01 | Records per class in the test set after cleaning (1,000) |
| `TEST_CANDIDATE_PER_CLASS` | 01 | Starting raw candidates per class (1,400). Raised automatically, per class, for any class that cannot fill its quota |
| `AUTO_EXPAND`, `MAX_ATTEMPTS`, `SAFETY` | 01 | `True` lets the candidate pool grow per class, at most `MAX_ATTEMPTS` = 6 times, with a `SAFETY` = 1.15 margin; `False` keeps 1,400 and warns instead |
| `CTGAN_EPOCHS` | 02 | 100, 300 or 500 |
| `EPOCHS` | 03–05 | 100, 300 or 500 — but see the note below on what an epoch means in each notebook |
| `CTGAN_BATCH`, `CTGAN_PAC` | 02 | 64 and 1 |
| `RESUME` | 02 | `True` reloads the classes an earlier session already finished and trains only the missing ones; `False` retrains all thirteen |
| `CKPT_ROOT` | 02 | Where the per-class checkpoints live. Must be on Google Drive — `/content` is wiped when the runtime dies |
| `TARGET_PER_CLASS` | 02–05 | Rows per class after augmentation (6,000) |
| `DEVICE_CONFIG` | 03–05 | `"GPU"` or `"CPU"` |
| `MODEL_TAG`, `EPOCHS` | 06, 08 | Which augmented run to read |
| `TRAINING_FILE` | 08 | Which of the three scenarios to run |
| `RECORD_CURVE` | 08 | `False` skips the training-side learning curve; results are unchanged |
| `TUNE` | 08 | `True` searches and writes `best_params.json`; `False` reads it back, keyed by classifier name |
| `USE_GPU` | 02, 08 | GPU for CTGAN and for XGBoost |

---

## Session plan

**Session A — CPU.** Notebook 01. Check the headroom table it prints: if the
tightest class has under 100 spare test candidates, raise
`TEST_CANDIDATE_PER_CLASS` and re-run. Download all three CSV files.

**Session B — GPU.** Notebooks 02–05, three times each (100, 300, 500 epochs).
24 files. This is the only stage where the GPU is genuinely required.

**Session C — CPU, at the same time as B.** Notebook 07. It needs only
`df_train.csv` and produces the 21-model ranking.

**Session D — GPU preferred.** Notebook 08 with `TUNE = True` and
`RECORD_CURVE = True` for the one configuration whose learning curve goes into
the paper. Then `TUNE = False` and `RECORD_CURVE = False` for the rest. Run the
class-weighting variant once on `df_train.csv` for the weighted baseline, and
re-run the chosen configurations against `df_unseen.csv`.

**Session E — CPU.** Notebook 06, once, for the winning model's
`_synthetic_only` file. Produces the KDE figure and the KS / Wasserstein table.

---

## Reproducibility

| Component | Seed |
|---|---|
| Python `random`, NumPy, `PYTHONHASHSEED` | 42 |
| PyTorch (CWGAN, TableGAN, MedGAN), with `cudnn.deterministic = True` | 42 |
| CTGAN per class | `42 + class index` |
| Train / test / unseen partition (notebook 01) | 42 |
| `StratifiedKFold(n_splits=5, shuffle=True)` | 42 |
| `RandomizedSearchCV(n_iter=5, scoring='f1_macro')` | 42 |
| LightGBM, XGBoost, Random Forest (notebook 08) | 42 |
| Notebook 07: internal split and all 21 estimators | **123** |

Notebook 08 prints this block at the start of a run, ready to paste into the
Reproducibility subsection.

---

## Things to keep in mind

**An "epoch" does not mean the same thing in 02, 03, 04 and 05.** In notebook 02
it is a conventional pass over the rows of one class, in mini-batches of 64. In
notebook 03 it is one generator update preceded by five critic updates, each on
a random mini-batch of 256 rows — 500 epochs is therefore about 16 effective
passes over `df_train`, not 500. In notebooks 04 and 05 it is a single
full-batch update. The three epoch settings are comparable **within** an
architecture and not **between** architectures, and the manuscript should say so
rather than presenting 100 / 300 / 500 as a shared axis. `GAN_architecture_table.docx`
states this per model.

**TableGAN and MedGAN can fall back to perturbed real rows.** When rejection
sampling cannot reach the quota for a class within the attempt limit, the
shortfall is filled by bootstrap resampling of that class's real records with
Gaussian noise σ = 0.005. Those rows end up in the `_synthetic_only` file. The
notebook prints how many were added per class — record that number, because it
bounds how much of the "synthetic" data is genuinely generated.

**Notebook 07 does not use `df_test`.** It splits `df_train` internally with
`random_state=123`. Its numbers are not comparable with notebook 08.

**Notebook 06 must read the `_synthetic_only` file.** Pointing it at the
combined file makes more than half the "synthetic" rows real records, which
pulls every KS and Wasserstein figure towards zero. The notebook checks the
overlap and warns.

**Only CTGAN is discrete-aware, and only partly.** Notebook 02 declares 23
discrete feature columns plus the target. `Neurological Assessments` is not
among them — it is listed under `continuous_columns` — so CTGAN generates it
continuously and the validity cell snaps it afterwards, exactly as for the other
three models. The before/after validity report each notebook prints is itself a
result worth reporting.

**Record counts.** Splitting before cleaning and holding out 1,300 raw records
changed the partition sizes: `df_train` is about 48.4k rows, not 50,401. Only
the 13,000 balanced test set is unchanged by design. Any table in the manuscript
that reports record counts must be refilled from the output of notebook 01.

---

## Notebook notes

These are the two long explanatory cells that used to sit inside the notebooks.
They are kept here so the notebooks themselves stay readable; nothing in them is
required to run the code.

### The unseen partition (notebook 01)

`UNSEEN_PER_CLASS` is set to **100**, which holds out 1,300 raw records — 100 per
class — before any cleaning. Set it to 0 to reproduce the two-partition design.

The reason a third partition is justified is **not** that `df_test` is
contaminated; after the leakage fix it is not. It is that `df_test` has been
*cleaned*: rows whose Waist Circumference or Pulmonary Function fell outside the
training IQR bounds were dropped, and so was every row holding a zero. Real
records arrive without that filter, so a model scored only on `df_test` is graded
on a tidier population than it would meet in use.

A partition that answers this has to be carved from the **raw** data, before any
cleaning. Taking rows out of `df_train` after cleaning would produce a set
exactly as tidy as `df_test` and would measure nothing new.

Two figures are then reported rather than one: performance under curated
conditions (`df_test`, balanced, 1,000 per class) and performance under realistic
conditions (`df_unseen`, untouched). The gap between them is itself a result.

Size governs what the second figure can support. At 1,300 records the 95% margin
of error on accuracy is roughly ±1.6 percentage points, wider than the spread
separating the configurations, so **use the unseen set for one honest headline
number, never to rank configurations**. Ranking stays with `df_test` and its
13,000 records, where the margin is about ±0.5 points. For reference: 401 records
would give about ±2.9 points, and 200 per class (2,600 records) about ±1.2.

### The class-weighted variant of notebook 08

This variant is for the two training sets that are **not** class-balanced:

| Scenario | File | Shape of the imbalance |
|---|---|---|
| 1 | `df_train.csv` | the original distribution, roughly 1,400 to 4,500 records per class |
| 3 | `<stem>_synthetic_only.csv` | the reverse — the scarce classes needed the most synthetic rows |

The combined file is uniform at 6,000 per class and does not belong here; run it
in the original notebook.

**What differs from the original.** Every class is weighted by the inverse of its
frequency, inside the hyperparameter search, inside each fold, and in the final
fit. LightGBM and Random Forest take this as `class_weight`; XGBoost, which has
no such parameter for multi-class problems, receives the equivalent per-row
sample weights. For the folds the weights are recomputed from the fold's own
distribution, so nothing crosses a fold boundary. Balanced accuracy is reported
alongside the existing metrics, and the class distribution and the resulting
weights are printed before anything is fitted.

The test partition is untouched: 13,000 records, 1,000 per class, never weighted.

Setting `CLASS_WEIGHT = False` reproduces the original, unweighted behaviour.
Running both once is worthwhile — the difference between them is a direct measure
of what the imbalance was costing.
