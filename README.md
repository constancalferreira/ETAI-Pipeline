# Constança Ferreira - 20260504

## Overview
This repository hosts the progressive development of an end-to-end predictive pipeline structured for the Exploratory Topics in Artificial Intelligence (ETAI) course. The primary objective is to predict two-year criminal recidivism (two_year_recid) using historical and demographic data of defendants from the COMPAS dataset, which is the real dataset investigated by ProPublica in 2016 to audit risk-assessment algorithms used in US courts.

## Dataset Variables

The table below outlines each variable in the COMPAS dataset, its role in the pipeline, its meaning, and how it is handled during data cleaning and preprocessing.

| Variable Name | Role in Pipeline | Description | Preprocessing Insights |
| :--- | :--- | :--- | :--- |
| **`two_year_recid`** | Target ($y$) | Binary indicator of whether the individual reoffended within two years ($1 = \text{Yes}, 0 = \text{No}$). | Used as the primary target label for all classification models and metric calculations. |
| **`id`** | Dropped (`drop_columns`) | Unique row/individual identifier. | Excluded from features (`drop_columns`) to prevent models from memorizing row IDs. |
| **`decile_score`** | Dropped (`drop_columns`) | COMPAS proprietary risk score decile (1 to 10). | Excluded from features (`drop_columns`) because it is COMPAS's own proprietary score; retained only for external fairness audits (false positive rate comparisons). |
| **`score_text`** | Dropped (`drop_columns`) | Risk category text ("Low", "Medium", "High"). | Excluded from features (`drop_columns`) and utilized strictly as a baseline benchmark in the fairness report to evaluate COMPAS's own predictions. |
| **`race`** | Sensitive Attribute | Demographic racial category of the individual (e.g., African-American, Caucasian, Hispanic, Asian, Native American, Other). | **Never used as a feature** (`sensitive_attr`). Designated exclusively for calculating group-level False Positive Rates in the fairness audit. |
| **`sex`** | Categorical Feature | Biological sex of the individual (Male, Female). | Canonicalized to standard capitalised spellings ("Male", "Female")|
| **`age_cat`** | Categorical Feature | Age bracket grouping (e.g., Less than 25, 25-45, Greater than 45). | Treated as a categorical feature, mapped, and encoded within the column transformer pipeline. |
| **`c_charge_degree`** | Categorical Feature | Degree of the current criminal charge (F = Felony, M = Misdemeanor). | Canonicalized via diagnostic rules, treated as a source for missingness flags (`c_charge_degree_was_missing`), and encoded. |
| **`age`** | Numeric Feature | Age of the individual in years. | Bounded by domain validity rules ($18 \le \text{age} \le 100$); handled via median imputation and robust scaling. |
| **`priors_count`** | Numeric Feature | Number of prior offenses recorded for the individual. | Cleaned from text tokens, bounded ($0 \le \text{priors\_count} \le 60$), triggered missingness indicator flags (`priors_count_was_missing`), and scaled. |
| **`juv_fel_count`** | Numeric Feature | Count of juvenile felony charges. | Validated against domain rules ($\ge 0$); imputed and scaled as a numeric feature. |
| **`juv_misd_count`** | Numeric Feature | Count of juvenile misdemeanor charges. | Imputed and scaled as part of the standard numeric feature set. |
| **`juv_other_count`** | Numeric Feature | Count of other miscellaneous juvenile charges. | Imputed and scaled as part of the numeric feature set. |
| **`prior_offenses`** | Redundant (`drop_columns`) | Legacy duplicate/redundant column tracking prior offenses. | Dropped automatically during `clean_dataset()` based on `diagnostic_config["redundant_columns"]`. |
| **`age_in_months`** | Redundant (`drop_columns`) | Age expressed in months. | Dropped automatically as a redundant column during dataset cleaning. |
| **`juvenile_total`** | Redundant (`drop_columns`) | Aggregated total of juvenile offenses. | Dropped automatically as a redundant column to prevent multicollinearity with individual juvenile counts. |

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| **3** | EDA Diagnostics & Preprocessing | Transition to **Evidence-Based Diagnostics**: statistical missingness tests (Chi-square and Cramér's V), domain rule validation, rigorous duplicate detection, and multicollinearity analysis. Implementation of a robust cleaning recipe, smart imputation with binary indicators for MNAR columns (`_was_missing`), and integration of a data-leakage-proof `ColumnTransformer`. |
| **4** | Preprocessing & Cross-Validation | Setting aside a locked 20% test set so we can evaluate models honestly via stratified 5-fold cross-validation instead of trusting a single split. Switched to scikit-learn's cross-fitting target encoder to prevent data leakage, cleaned up categorical placeholder bugs (like `"nan"` strings), decoupled de-duplication so new inference data works smoothly, and added dummy and random forest benchmarks. |


## Best Model

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | — | — | — |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.672 ± 0.017 (+0.013)  | 0.679 | {'model__max_depth': 6, 'model__min_samples_leaf': 101, 'model__criterion': 'gini'}|
| Random forest | 0.650 ± 0.019 (+0.082)  | 0.681 ± 0.015 (+0.009) | 0.683 | {'model__max_depth': 22, 'model__min_samples_leaf': 42, 'model__max_features': 0.5} |
| Oblique random forest | 0.647 ± 0.011 (+0.114) | 0.679 ± 0.013 (+0.013) | 0.683 | {'model__estimator__tree__max_depth': 6, 'model__estimator__tree__min_samples_leaf': 3, 'model__estimator__project__n_components': 5} |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.017 (+0.004) | 0.684 | {'model__estimator__max_depth': 4, 'model__n_estimators': 220, 'model__learning_rate': 0.025081156860452335} |

### Conclusion:

AdaBoost achieves the highest honest validation accuracy at 0.682, closely followed by Random Forest at 0.681.  The performance lead of AdaBoost over Random Forest is 0.001. Because this lead is vastly smaller than the fold-to-fold standard deviation (0.017), the difference is insignificant. For all practical purposes, AdaBoost and Random Forest tie.

The optimized hyperparameters align well with model theory: AdaBoost balanced complex feature interactions by combining a tree slightly deeper than usual (max_depth=4) with a small learning rate (0.025) and 220 estimators, while Random Forest relied on low-bias, deeper trees whose variance is safely managed through leaf constraints (min_samples_leaf=42) and feature subsampling (max_features=0.5).

Ultimately, I would report AdaBoost's honest nested cross-validation accuracy of 0.682 (± 0.017) as my final result because nested cross-validation provides an honest evaluation that prevents data leakage and tuning optimism, giving me a reliable estimate of how my pipeline will generalize.

## Feature Engineering
Three engineered features were implemented:
* **`total_contacts`**: Total lifetime interactions (`priors_count + juv_fel_count + juv_misd_count + juv_other_count`).
* **`felony_concentration`**: Proportion of juvenile felonies relative to total contacts.
* **`priors_per_adult_year`**: Adult criminal velocity (`priors_count / (age - 17)`).

All three features rely entirely on deterministic arithmetic and fixed thresholds with zero parameters learned from training labels, making them completely free of data leakage and safe to compute before splitting.

### Results

| Model Family | Baseline Nested CV Accuracy | Enhanced Nested CV Accuracy | Delta | Fold Std |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.672 | **0.675** | $+0.003$ | $\sim 0.013$ |
| **Random Forest** | 0.681 | **0.676** | $-0.005$ | $\sim 0.012$ |
| **AdaBoost** | 0.682 | **0.681** | $-0.001$ | $\sim 0.012$ |
| **Decision Tree** | 0.672 | **0.673** | $+0.001$ | $\sim 0.010$ |

---

* Logistic Regression saw a slight positive shift ($+0.003$), while tree-based models showed negligible or slightly negative performance deltas ($[-0.005, -0.001]$).
* All performance deltas ($0.001$ to $0.005$) are substantially smaller than the fold-to-fold standard deviation ($\sim 0.012$), meaning the variations are statistically insignificant.
* **Conclusion:** Tree models use axis-parallel splits that can implicitly approximate these sums and ratios from raw features (`priors_count`, `juv_fel_count`, `age`), while Logistic Regression natively handles linear combinations. Explicitly engineering them introduced feature redundancy rather than new predictive boundaries.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
