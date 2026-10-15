# Constança Ferreira - 20260504

## Overview
This repository hosts the progressive development of an end-to-end predictive pipeline structured for the Exploratory Topics in Artificial Intelligence (ETAI) course. The primary objective is to predict two-year criminal recidivism (two_year_recid) using historical and demographic data of defendants from the COMPAS dataset, which is the real dataset investigated by ProPublica in 2016 to audit risk-assessment algorithms used in US courts.

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
