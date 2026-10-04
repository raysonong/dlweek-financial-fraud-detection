# Financial Fraud Detection — Track 2

A notebook-based financial fraud detection project that assigns each transaction a fraud probability using a weighted blend of XGBoost and CatBoost.

## Project files

- `TeamName_Track2_Notebook.ipynb`: data exploration, feature engineering, validation, training, and prediction.
- `requirements.txt`: pinned versions of the modelling libraries, notebook kernel, and supporting dependencies.
- `model.pkl`: generated model bundle and feature list (ignored by Git).
- `predictions.csv`: generated submission IDs and probabilities (currently not ignored by Git).

The notebook expects the training and testing datasets described below. CSV files are currently not ignored by Git; whether they are included in a checkout depends on which files were committed.

## Installation

Install Python 3.12 first, then run the commands below from the project directory. The default setup uses Python's built-in `venv` and pip; `uv` is optional.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is unavailable but `python --version` reports Python 3.12, use `python -m venv .venv` for the first command.

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Windows Command Prompt (cmd)

```cmd
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.12` is unavailable but `python3 --version` reports Python 3.12, use `python3 -m venv .venv`. Some Linux distributions require installing their Python 3.12 venv package before creating the environment.

If `.venv` already exists with Python 3.12, skip its creation and activate it. If pip is missing from an existing environment, run `python -m ensurepip --upgrade` using that environment's Python before the pip commands.

### Open the notebook

For VS Code, install the Python and Jupyter extensions, open `TeamName_Track2_Notebook.ipynb`, and choose **Select Kernel > Python Environments > .venv**. `ipykernel` is already included in `requirements.txt`.

Alternatively, install and start JupyterLab from the activated environment:

```bash
python -m pip install jupyterlab
python -m jupyterlab
```

If you skipped activation on Windows, replace `python` with `.\.venv\Scripts\python.exe`. JupyterLab is optional and is not included in `requirements.txt`.

### Optional alternative: uv

If you already have `uv` installed, it can create the environment and install the same requirements without pip:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

Select `.venv` as the notebook kernel as described above.

## Dataset setup

Place both CSV files in the project root and start Jupyter from that directory. The notebook loads these exact filenames:

```text
training_dataset.csv
testing_dataset.csv
```

For example:

```text
<project directory>/
├── TeamName_Track2_Notebook.ipynb
├── requirements.txt
├── training_dataset.csv
└── testing_dataset.csv
```

Use these exact filenames. The notebook does not search for alternative names or use an environment-variable override.

The training CSV must contain `id`, `fraud`, and the input columns below. The testing CSV must contain `id` and the same input columns; fraud labels are not required. The notebook copies the test IDs into the submission.

| Input column | Purpose |
| --- | --- |
| `transaction_amount` | Transaction size |
| `transaction_hour` | Hour of day |
| `transactions_last_24h` | Recent daily transaction count |
| `spend_last_24h` | Recent daily spending |
| `account_age` | Account age |
| `new_device` | New-device indicator, encoded as 0 or 1 |
| `transactions_last_1h` | Recent hourly transaction count |
| `merchant_category` | Merchant category |
| `country` | Country category |
| `transaction_channel` | Transaction channel category |

`fraud` must be encoded as `0` for legitimate transactions and `1` for fraud. Numerical inputs should be numeric; missing values are handled by the models and preprocessing.

## Running the notebook

1. Open `TeamName_Track2_Notebook.ipynb` and select the environment created above.
2. Run all cells in order. Cross-validation trains both models across 25 folds, so runtime depends on dataset size and hardware.
3. Review the validation tables and plots.
4. Find `model.pkl` and `predictions.csv` in the working directory after the training and prediction cells finish.

Install the pinned dependencies from `requirements.txt` before running the notebook. The notebook does not install packages automatically.

**File behavior:** rerunning the notebook overwrites `model.pkl` and `predictions.csv`. The existing `requirements.txt` remains unchanged.

## How it works

1. **Load and explore:** read labelled training transactions, remove `id` and `fraud` from model inputs, and inspect class imbalance, missing values, and fraud associations.
2. **Engineer four features:** multiply `new_device` by transaction amount, hourly activity, and daily activity; represent time of day with `cos(2π × transaction_hour / 24)`.
3. **Prepare each model:** XGBoost receives numerical features with missing values preserved and categorical features imputed to `Unknown` and one-hot encoded. CatBoost receives numerical features and categorical strings, with missing categories replaced by `Unknown`.
4. **Validate:** use stratified five-fold cross-validation for seeds `42`, `123`, `2026`, `7`, and `99`. Each fold trains both models on the same training rows and predicts the held-out rows. Model randomness stays fixed at seed `42`.
5. **Blend:** compute `0.70 × XGBoost probability + 0.30 × CatBoost probability`.
6. **Train and save:** refit both models on all labelled rows and save them with weights, threshold, category names, and the feature list using joblib.
7. **Predict and export:** apply the same feature engineering to test transactions, blend predictions, and write `id` and `prediction` columns to `predictions.csv` in the original test-row order.

Average Precision (AP) is the main ranking metric because fraud is rare. The notebook reports AP as the average fold score, averaged across the five seeds. Precision, recall, and F1 are calculated from each seed's out-of-fold predictions using a `0.20` threshold. Submission values remain probabilities between 0 and 1.

The notebook's written summary reports AP **23.12%**, precision **44.85%**, recall **20.34%**, and F1 **27.98%**. These are recorded results, not independently reproduced here. The model-comparison table is hard-coded; running the notebook retrains and validates the final blend, but does not repeat the earlier model comparison or feature-subset search.

## Outputs and troubleshooting

- **Missing CSV / `FileNotFoundError`:** check that `training_dataset.csv` and `testing_dataset.csv` exist in the notebook working directory with these exact names.
- **Missing column / `KeyError`:** check the required column names and CSV headers.
- **Import errors:** select the correct notebook kernel and install requirements into that same environment.
- **Pinned dependency installation fails:** read pip's error for the failing package and Python/platform compatibility. If you deliberately adjust a version, update `requirements.txt` manually and rerun validation.

`model.pkl` is a joblib bundle, not a standalone predictor: inference also needs the notebook's feature engineering and categorical preparation. The test set has no labels, so the notebook cannot calculate test-set performance.

The `.gitignore` excludes generated models, virtual environments, caches, and local secrets. It currently allows CSV files, including datasets and `predictions.csv`. Keep the notebook and `requirements.txt` under version control. If a competition submission requires `model.pkl`, package it separately from Git along with any other required deliverables.
