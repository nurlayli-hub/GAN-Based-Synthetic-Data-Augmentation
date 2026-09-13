# Pipeline Reference

Every notebook: what goes in, what it does, what comes out, and whether it needs
a GPU.

---

## Run order

```
STAGE 1                 STAGE 2 (parallel)              STAGE 3        STAGE 4       STAGE 5
┌──────────────┐    ┌──────────────────────────┐   ┌───────────┐  ┌──────────┐  ┌──────────┐
│      01      │    │  02  03  04  05          │   │    08     │  │    09    │  │    06    │
│ preprocessing│───▶│  CTGAN CWGAN Table Med   │──▶│classifier │─▶│ compare  │─▶│ KDE + KS │
│              │    │                          │   │           │  │ pick best│  │ + WD     │
│              │    │  07  LazyClassifier      │   │           │  │          │  │          │
└──────────────┘    └──────────────────────────┘   └───────────┘  └──────────┘  └──────────┘
   CPU                  GPU          CPU               GPU opt        GPU opt       CPU
```

**Stage 2 runs in two independent sessions.** Notebooks 02–05 and notebook 07
both read `df_train.csv` and produce outputs that never touch each other: 07
only screens 21 baseline models to justify picking LightGBM, XGBoost and Random
Forest, while 02–05 generate synthetic data. Run 07 on a **CPU runtime** so it
does not consume GPU quota.

**Stage 5 comes last on purpose.** Notebook 06 is run once, for the single
generative model that stage 4 identified as best, to produce the KDE figure and
the KS / Wasserstein table that go into the paper.

---

## GPU

| Notebook | GPU | Why |
|---|---|---|
| **01** preprocessing | **Not needed** | pandas and scikit-learn only |
| **02** CTGAN | **Required** | `enable_gpu=True`; the cell asserts the generator is on CUDA |
| **03** CWGAN | **Required** | PyTorch, WGAN-GP with 5 critic iterations per step |
| **04** TableGAN | **Required** | PyTorch, plus rejection sampling in batches of 50,000 |
| **05** MedGAN | **Required** | PyTorch, autoencoder + latent GAN |
| **06** quality | **Not needed** | scipy and seaborn |
| **07** LazyClassifier | **Not needed** | 21 scikit-learn models, all CPU |
| **08** classifier | **Optional** | Only XGBoost uses it (`USE_GPU`). LightGBM and Random Forest are CPU either way |
| **09** comparison | **Optional** | Same as 08 |

Practical consequence: **only stage 2's augmentation half genuinely needs the
GPU.** Notebook 07 can run at the same time on a separate CPU session. Stages 3
and 4 benefit from a GPU but will finish without one.

---

## Master table

| # | Notebook | Input | What it does | Output | GPU |
|---|---|---|---|---|---|
| **01** | `01_data_preprocessing.ipynb` | Raw Kaggle CSV, 70,000 x 34 | Duplicate and NaN check → **partition the raw data first** → fit IQR bounds on the training pool only, apply to both partitions → drop rows holding a zero → encode (3 ordinal, 16 binary, 1 one-hot group, LabelEncoder on the target) → Min-Max scale, fitted on training only. Prints a leakage audit and a per-class headroom table. | `df_train.csv`<br>`df_test.csv` (13,000, 1,000/class)<br>`df_unseen.csv` (1,300, raw) | No |
| **02** | `02_augmentation_ctgan.ipynb` | `df_train.csv` | Checkpoints each class to Drive as it finishes, so a Colab disconnect costs only the class in progress. One CTGAN **per class**, conditional, `discrete_columns` declared, pac=1, to 6,000 rows per class. Asserts the generator is on CUDA. Option A validity check. | `ctgan_perclass_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **03** | `03_augmentation_cwgan.ipynb` | `df_train.csv` | Conditional WGAN-GP, label one-hot (13), λ=10, 5 critic iterations. Generates directly for the target class. Option A correction. | `cwgan_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **04** | `04_augmentation_tablegan.ipynb` | `df_train.csv` | Unconditional GAN + rejection sampling (50,000 per batch, up to 50 attempts), bootstrap + Gaussian noise (σ=0.005) for any shortfall. Option A correction. | `tablegan_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **05** | `05_augmentation_medgan.ipynb` | `df_train.csv` | Autoencoder + latent-space GAN, unconditional, rejection sampling with the same fallback. Option A correction. | `medgan_gpu_epoch<E>_augment6000_seed42.csv`<br>`..._synthetic_only.csv` | Yes |
| **07** | `07_baseline_model_screening.ipynb` | `df_train.csv` | **LazyClassifier** over 21 baseline models on its own internal split (`test_size=0.2, random_state=123` — it does **not** use `df_test`). Ranks by macro F1. | `hasil_evaluasi_model_fulldata.csv` on Drive; the ranking that justifies the three classifiers | No |
| **08** | `08_classification_and_evaluation.ipynb` | one training set (`TRAINING_FILE`) + `df_test.csv` | Deep single-run notebook: cached tuning, stratified 5-fold CV, converged learning curves, confusion matrix, per-class report, accuracy cross-check, separation audit. | `learning_curve_lightgbm.png`<br>`learning_curve_xgboost.png`<br>`learning_curve_random_forest.png`<br>`best_params.json` | Optional |
| **09** | `09_scenario_and_model_comparison.ipynb` | `df_train.csv`, `df_test.csv`, `df_unseen.csv`, every `<stem>.csv` and `<stem>_synthetic_only.csv` | Every training set against the same test set with every classifier, tuning once on the baseline. **McNemar**, per-class recall, separation audit, accuracy cross-check, and curated-vs-realistic comparison on `df_unseen`. | `09_results.csv`<br>`09_mcnemar.csv`<br>`09_per_class_recall.csv`<br>`09_separation_audit.csv`<br>`09_curated_vs_realistic.csv`<br>`09_predictions.csv`<br>`09_unseen_predictions.csv` | Optional |
| **06** | `06_synthetic_data_quality_v2.ipynb` | `df_train.csv` + **`<stem>_synthetic_only.csv`** of the winning model | Equal-size sampling, KDE overlays, **Kolmogorov–Smirnov test** and **Wasserstein distance** per continuous variable. Warns if the synthetic file still contains real records. | KDE figure (Figure 2) and the KS / Wasserstein table (Table 8) | No |

---

## The three scenarios

Notebooks 02–05 each write **two** files, so three training scenarios can be
compared against one untouched test set:

| Scenario | Training set | File | Question it answers |
|---|---|---|---|
| 1 | Original only | `df_train.csv` | What does the data give with no augmentation? |
| 2 | Original + synthetic | `<stem>.csv` | Does augmentation add anything? |
| 3 | Synthetic only | `<stem>_synthetic_only.csv` | Did the generator learn the real structure at all? |

Notebook 09 runs all three. Notebook 08 runs whichever one `TRAINING_FILE`
points at.

---

## The three partitions

| Partition | Rows | Condition | Use |
|---|---|---|---|
| `df_train.csv` | ~50k | cleaned, imbalanced | training, and the input to augmentation |
| `df_test.csv` | 13,000 | cleaned, balanced 1,000/class | **ranks** the configurations |
| `df_unseen.csv` | 1,300 | **raw, never cleaned** | one honest real-world number |

`df_unseen` is held out from the raw data **before** any cleaning, so it still
contains the outliers and zero-coded missing values that `df_test` had removed.
Notebook 09 scores it with the same fitted models and reports the gap.

At 1,300 records the 95% margin of error on accuracy is about ±1.6 percentage
points — wider than the spread between configurations. Use it for one headline
figure, never for ranking.

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
one another.

---

## Switches

| Switch | Notebook | Meaning |
|---|---|---|
| `UNSEEN_PER_CLASS` | 01 | **100** — holds out 1,300 raw records before cleaning. 0 reproduces the two-partition design |
| `TEST_CANDIDATE_PER_CLASS` | 01 | Starting raw candidates per class (1,400). Raised automatically, per class, for any class that cannot fill its quota after cleaning |
| `AUTO_EXPAND` | 01 | `True` lets the candidate pool grow per class; `False` keeps 1,400 and warns instead |
| `EPOCHS` / `CTGAN_EPOCHS` | 02–05 | 100, 300 or 500 |
| `RESUME` | 02 | `True` reloads the classes an earlier session already finished and trains only the missing ones; `False` retrains all thirteen |
| `CKPT_ROOT` | 02 | Where the per-class checkpoints live. Must be on Google Drive — `/content` is wiped when the runtime dies |
| `TARGET_PER_CLASS` | 02–05 | Rows per class after augmentation (6,000) |
| `MODEL_TAG`, `EPOCHS` | 06, 08 | Which augmented run to read |
| `TRAINING_FILE` | 08 | Which of the three scenarios to run |
| `RECORD_CURVE` | 08 | `False` skips the training-side learning curve; results are unchanged |
| `TUNE` | 08, 09 | `True` searches and stores; `False` reuses `best_params*.json` |
| `USE_GPU` | 02–05, 08, 09 | GPU for the GANs and for XGBoost |
| `RUNS`, `CLASSIFIERS` | 09 | Which models and classifiers to include |

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
the paper. Then `TUNE = False` and `RECORD_CURVE = False` for the rest.

**Session E — GPU preferred.** Notebook 09. Start with a short `RUNS` list, add
the rest once it works. Produces the comparison table, McNemar, per-class recall
and the curated-vs-realistic gap.

**Session F — CPU.** Notebook 06, once, for the winning model's
`_synthetic_only` file. Produces the KDE figure and the KS / Wasserstein table.

---

## Reproducibility

| Component | Seed |
|---|---|
| Python `random`, NumPy, `PYTHONHASHSEED` | 42 |
| PyTorch (CWGAN, TableGAN, MedGAN) | 42 |
| CTGAN per class | `42 + class index` |
| Train / test / unseen partition (notebook 01) | 42 |
| `StratifiedKFold(n_splits=5, shuffle=True)` | 42 |
| `RandomizedSearchCV(n_iter=5, scoring='f1_macro')` | 42 |
| LightGBM, XGBoost, Random Forest | 42 |
| LazyClassifier internal split (notebook 07) | **123** |

Notebooks 08 and 09 print this block at the start of a run, ready to paste into
the Reproducibility subsection.

---

## Things to keep in mind

**Notebook 07 does not use `df_test`.** It splits `df_train` internally with
`random_state=123`. Its numbers are not comparable with 08 or 09, and Figure 1
should not claim otherwise.

**Notebook 06 must read the `_synthetic_only` file.** Pointing it at the
combined file makes more than half the "synthetic" rows real records, which
pulls every KS and Wasserstein figure towards zero. The notebook now checks and
warns.

**Record counts changed.** The leakage fix in notebook 01 estimates the cleaning
rule from a smaller pool, and `UNSEEN_PER_CLASS = 100` removes another 1,300
rows before cleaning, so 63,401 / 50,401 no longer hold. Only the 13,000 balanced
test set is unchanged by design. Tables 2 and 4 of the manuscript must be
refilled from the new output.

**Only CTGAN is discrete-aware.** The other three end in a `Sigmoid` and emit
continuous values for all 36 columns. The Option A correction restores legal
values; the before/after validity report it prints is itself a result worth
reporting.
