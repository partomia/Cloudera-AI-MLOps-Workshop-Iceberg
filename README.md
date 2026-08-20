# Credit Risk Model — Loan Default Prediction

An end-to-end machine learning project that predicts the probability of a borrower defaulting on a loan. Trained on real loan data from an Iceberg gold table via Impala, built with XGBoost, and served as a REST API via Flask — designed to run on **Cloudera AI (CML) / AI Workbench**.

---

## Project Overview

Credit risk scoring is a core use case in financial services. This project demonstrates:

- Loading the M4 gold feature table (`federal12_gold.credit_risk_features`) from Iceberg via Impala
- Binary classification model training (default vs. no default) using XGBoost, with class imbalance handled via `scale_pos_weight`
- MLflow experiment tracking integrated with CML's native Experiments UI
- Model serialization and deployment as a REST API
- Automated CI/CD pipeline with Gini/KS validation gates on every push

### Features used for prediction

The model uses 21 features, defined once in `features.py` and imported everywhere else so the training and serving contracts can't drift apart.

| Group | Features |
|---|---|
| Loan terms | `principal`, `tenure_months`, `interest_rate`, `emi_amount` |
| Borrower | `annual_income`, `employment_years`, `age_years` |
| Bureau | `credit_score`, `num_credit_lines`, `num_delinquencies`, `debt_to_income`, `credit_history_months`, `credit_utilisation`, `is_ntc` |
| Derived / behavioural | `loan_to_income`, `emi_to_income_monthly`, `prior_loan_count`, `prior_default_count`, `prior_principal_sum` |
| Categorical | `employment_status`, `purpose` |

**Bureau fields are `null` for new-to-credit (NTC) borrowers** (4,449 of 17,636 rows) — left as `NaN` on purpose. XGBoost splits on missingness natively, and "no bureau file" is itself predictive, so these are never imputed.

Excluded on purpose: `loan_id`/`customer_id` (identifiers), `disbursal_date` (audit only), `_gold_ts` (pipeline audit column), `state` (too high-cardinality for the number of test defaults available).

---

## Repository Structure

```
├── requirements.txt                  # Python dependencies
├── cdsw-build.sh                     # CML/CDSW build script — installs deps, trains model
├── features.py                       # Single source of truth: feature list, payload validation, decision threshold
├── 01_load_gold.py                   # Load the M4 gold table from Iceberg via Impala -> loan_data.csv
├── 02_train_model.py                 # Train XGBoost model, log to MLflow, save artifacts
├── 03_predict.py                     # Flask/Gunicorn REST API — session-based serving
├── 04_test_api.py                    # Test the running Flask API with real applicants from the gold table
├── 05_validate_model.py              # Gini/KS validation gate — used by CI/CD pipeline
├── diagnose.py                       # Outlier scan + pulls real low/high/NTC applicants for testing
├── cml_model.py                      # CML Model Deployment entry point (production path)
└── .github/workflows/retrain.yml     # GitHub Actions CI/CD workflow
```

<img width="735" height="367" alt="image" src="https://github.com/user-attachments/assets/d54f39f4-3ac7-4492-bea1-27eab6ee25e1" />

---

## MLflow Experiment Tracking

This project uses **MLflow** to log model parameters, KPIs, and artifacts. In Cloudera AI the `MLFLOW_TRACKING_URI` environment variable is automatically set in every session — no configuration needed. When running locally, runs are stored in `./mlruns`.

Each training run (`02_train_model.py`) records:

| What is logged | MLflow key |
|---|---|
| Hyperparameters | `n_estimators`, `max_depth`, `learning_rate`, `eval_metric`, `scale_pos_weight` |
| Dataset stats | `train_samples`, `test_samples`, `default_rate_portfolio`, `default_rate_holdout`, `n_features` |
| **KPI / success metrics** | `roc_auc`, `gini`, `ks`, `ks_threshold`, `accuracy` |
| Default class metrics | `precision_default`, `recall_default`, `f1_default` |
| Model artifact | XGBoost model (with signature + input example) |
| Encoder artifact | `label_encoders.pkl` |

> MLflow is pre-installed in CML sessions via `mlflow-cml-plugin`. It is intentionally excluded from `requirements.txt` to avoid breaking CML's pinned version. When it isn't installed (e.g. GitHub Actions), training and validation skip it automatically.

**To view experiment results in CML:** navigate to **Experiments** in the left sidebar of your project.

---

## Using This Project in Cloudera AI (CML)

### Prerequisites

- Access to a Cloudera AI (CML) workspace
- AI Workbench enabled on your CML cluster
- Git credentials configured in your CML profile
- A CML Data Connection named `federal-impala-1` with read access to `federal12_gold.credit_risk_features`

---

### Step 1 — Create a New Project from Git

1. Log in to your **Cloudera AI** workspace.
2. Click **New Project** on the Projects page.
3. Choose **Git** as the project source.
4. Enter the repository URL.
5. Set the project name and click **Create Project**.

---

### Step 2 — Open a Session and Install Dependencies

1. Inside the project, click **New Session**.
2. Select:
   - **Editor**: Workbench (or JupyterLab)
   - **Kernel**: Python 3
   - **Resource Profile**: At least 2 vCPU / 4 GB RAM
   - **Data Connection**: `federal-impala-1` (needed by Step 3)
3. Click **Start Session**.
4. In the **Terminal** tab, run:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 3 — Load the Gold Table

```bash
python 01_load_gold.py
```

Reads `federal12_gold.credit_risk_features` via Impala and writes `loan_data.csv`. This step needs a live Data Connection and Kerberos credentials, so it only runs inside a CML session — not in `cdsw-build.sh` and not in GitHub Actions.

Expected output:
```
Loaded 17,636 rows from federal12_gold.credit_risk_features
Features     : 21
Default rate : 4.22%
Nulls        : ... (bureau fields for new-to-credit borrowers — left as NaN on purpose)
```

---

### Step 4 — Train the Model

```bash
python 02_train_model.py
```

Trains an XGBoost classifier, logs the run to MLflow, and saves:
- `credit_risk_model.pkl` — trained model
- `label_encoders.pkl` — encoders for `employment_status` and `purpose`

Prints a classification report, ROC-AUC/Gini/KS, top feature importances, risk-quintile lift, and a new-to-credit vs. bureau-backed segment breakdown.

---

### Step 5 — Start the Prediction API

Run the API server from your **session terminal**:

```bash
python 03_predict.py
```

Gunicorn starts on `CDSW_APP_PORT` (falls back to 5000, then probes 5001/9090/9091 if unavailable):

```
=== CML PORT DIAGNOSTICS ===
  ...
  Binding on port: 5000
============================
[INFO] Listening at: http://0.0.0.0:5000
```

Keep this terminal open — the server must stay running for Step 6.

---

### Step 6 — Test the API

```bash
python 04_test_api.py
```

Sends three **real applicants pulled from the gold table** (not hypothetical profiles) — a low-risk bureau-backed loan, a high-risk bureau-backed loan with a prior default, and a new-to-credit loan with no bureau file at all — and compares the prediction against the loan's actual outcome. If the server bound to a non-default port, run with `API_PORT=<port> python 04_test_api.py`.

Expected output:
```
Health check: {'status': 'ok', 'features': 21}

[Low risk, bureau-backed]
  Default probability : 0.03xx
  Decision            : LOW (threshold 0.3793)
  Actual outcome      : repaid  -> correct

[High risk, prior default]
  Default probability : 0.8xxx
  Decision            : HIGH (threshold 0.3793)
  Actual outcome      : DEFAULTED  -> correct

[New-to-credit, no bureau file]
  Default probability : 0.9xxx
  Decision            : HIGH (threshold 0.3793)
  Actual outcome      : DEFAULTED  -> correct
```

Need more real payloads to test with? `python diagnose.py` scans for outliers and prints ready-to-use JSON for the lowest/highest-scoring bureau-backed loans and the highest-scoring new-to-credit loan.

---

### API Reference

#### `POST /predict`

**Request body (JSON)** — all 21 fields in `features.py` are required; bureau fields may be `null` for new-to-credit applicants:
```json
{
  "principal": 139938.0, "tenure_months": 36.0, "interest_rate": 0.2392,
  "emi_amount": 5484.0, "annual_income": 2740000.0, "employment_years": 7.0,
  "age_years": 34.0, "credit_score": 775.0, "num_credit_lines": 5.0,
  "num_delinquencies": 0.0, "debt_to_income": 0.14,
  "credit_history_months": 47.0, "credit_utilisation": 0.47, "is_ntc": 0.0,
  "loan_to_income": 0.0511, "emi_to_income_monthly": 0.024,
  "prior_loan_count": 2.0, "prior_default_count": 0.0,
  "prior_principal_sum": 4880000.0,
  "employment_status": "self_employed", "purpose": "business"
}
```

**Response:**
```json
{
  "default_probability": 0.0312,
  "prediction": 0,
  "risk_label": "LOW",
  "threshold": 0.3793
}
```

`prediction`/`risk_label` are `1`/`"HIGH"` when `default_probability >= threshold`. The threshold (`DECISION_THRESHOLD` in `features.py`) is the KS-optimal cut-off from training, **not 0.5** — the model is trained with `scale_pos_weight`, so raw probabilities are inflated relative to the ~4.22% portfolio base rate. They rank borrowers well but aren't calibrated, so the cut-off is a credit policy decision, not a modelling one.

#### `GET /health`

Returns `{"status": "ok", "features": 21}` when the API is running.

---

## CML Model Deployment (Production Path)

CML Model Deployments provide a managed, authenticated REST endpoint that runs 24/7 without needing an open session.

### Session API vs Model Deployment

| | Session API (`03_predict.py`) | Model Deployment (`cml_model.py`) |
|---|---|---|
| Lifecycle | Lives only while your session is open | Runs independently, always-on |
| Auth | None (open within session) | Access key required on every request |
| Scaling | Single process | Configurable replicas |
| Entry point | Flask route `POST /predict` | Plain Python function `predict(args)` |
| Port management | Manual (Gunicorn) | Handled entirely by CML |

`cml_model.py` handles both the classic CDSW runtime (parsed dict) and the newer PBJ runtime (JSON string, sometimes the whole request envelope) so the same file works on either.

### Step A — Create CML Jobs for the Pipeline

In CML → **Jobs → New Job**, create each job:

| Job name | Script | Dependency |
|---|---|---|
| `Load Gold Table` | `01_load_gold.py` | — |
| `Train Credit Risk Model` | `02_train_model.py` | Load Gold Table |
| `Validate Model KPIs` | `05_validate_model.py` | Train Credit Risk Model |

Run **Job 1** then **Job 2** in order. This produces `credit_risk_model.pkl` and `label_encoders.pkl` in the project filesystem and logs the run to the **Experiments** tab.

> `cdsw-build.sh` also runs `02_train_model.py` during the project build (against whatever `loan_data.csv` is already committed/present), so a fresh build already has model artifacts baked in even before you run the Jobs above. It does **not** run `01_load_gold.py` during the build — that step needs a live Data Connection that the Docker build environment doesn't have.
>
> The `.pkl` files must exist in the project before deploying the model.

### Step B — Deploy the Model

1. Go to **Models** → **New Model**
2. Fill in the configuration:

   | Field | Value |
   |---|---|
   | Name | `credit-risk-model` |
   | Description | XGBoost loan default predictor |
   | File | `cml_model.py` |
   | Function | `predict` |
   | Kernel | Python 3 |
   | CPU | 1 |
   | Memory | 2 GB |
   | Replicas | 1 |

3. Under **Example Input**, paste the same payload shown in [API Reference](#api-reference) above.
4. Click **Deploy Model** and wait for the status badge to turn **green (Running)**.

### Step C — Test the Deployed Model

#### From the CML UI

Model → **Test** tab → example input is pre-filled → click **Run**.

#### Via curl

```bash
curl -X POST https://<your-cml-workspace>/api/v1/projects/<username>/<project>/models/<model-id>/predict \
  -H "Content-Type: application/json" \
  -d '{
    "accessKey": "<your-model-access-key>",
    "request": {
      "principal": 139938.0, "tenure_months": 36.0, "interest_rate": 0.2392,
      "emi_amount": 5484.0, "annual_income": 2740000.0, "employment_years": 7.0,
      "age_years": 34.0, "credit_score": 775.0, "num_credit_lines": 5.0,
      "num_delinquencies": 0.0, "debt_to_income": 0.14,
      "credit_history_months": 47.0, "credit_utilisation": 0.47, "is_ntc": 0.0,
      "loan_to_income": 0.0511, "emi_to_income_monthly": 0.024,
      "prior_loan_count": 2.0, "prior_default_count": 0.0,
      "prior_principal_sum": 4880000.0,
      "employment_status": "self_employed", "purpose": "business"
    }
  }'
```

Response:
```json
{
  "success": true,
  "response": {
    "default_probability": 0.0312,
    "prediction": 0,
    "risk_label": "LOW",
    "threshold": 0.3793
  }
}
```

### Retraining and Redeployment

1. Re-run **Job 1** (Load Gold Table) if you want the latest data, then **Job 2** (Train Credit Risk Model) — new `.pkl` files are written and a new MLflow run is logged
2. Deployed model → **Builds** → **Rebuild** — CML picks up the new `.pkl` files and redeploys with zero downtime

---

## CI/CD Pipeline (GitHub Actions)

Every push to `main` retrains and validates the model against the **committed** `loan_data.csv`.

```
Push to main
     ↓
GitHub Actions fires (.github/workflows/retrain.yml)
     ↓
Step 1: pip install -r requirements.txt
     ↓
Step 2: python 02_train_model.py     → credit_risk_model.pkl
     ↓
Step 3: python 05_validate_model.py
     ↓
✅ Gini ≥ 0.35 AND KS ≥ 25 → pipeline green
❌ Either threshold missed  → pipeline red
```

No secrets or external services required — the pipeline runs entirely within GitHub Actions.

`loan_data.csv` is committed to the repo (see `.gitignore`) rather than regenerated in CI: refreshing it requires `01_load_gold.py`, which reads Iceberg via Impala and needs a live CML Data Connection and Kerberos credentials that GitHub Actions doesn't have. Refresh the data from inside a CML session (Step 3 above) and commit the updated CSV when you want CI to train on newer data.

### KPI thresholds

Defined at the top of `05_validate_model.py`:

```python
GINI_MIN = 0.35   # industry floor for a usable retail scorecard
KS_MIN   = 25.0   # separation between good and bad distributions
```

Gini and KS are used instead of ROC-AUC/F1 because the portfolio's default rate is only ~4.22% — an F1 gate on the default class would fail a perfectly good model, and a naive high-accuracy model that catches zero defaults would pass. Gini and KS measure ranking ability, which is what a credit decisioning system actually uses.

There's also a **non-blocking** segment check: new-to-credit borrowers are ~25% of the portfolio and default at 2.4x the overall rate, so a portfolio-level Gini can pass while that segment fails on its own (`SEGMENT_GINI_MIN = 0.30`, reported as a warning, not yet gating — see `SEGMENT_GATE_BLOCKING` in the script).

Thresholds sit below the observed bootstrap range, not at the median, so ordinary seed variation doesn't turn the pipeline red. Revisit them when switching to a different or larger dataset.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy<2.0` | Numerical operations (pinned for scikit-learn 1.3.0 compatibility) |
| `scikit-learn==1.3.0` | Preprocessing, metrics |
| `xgboost==1.7.6` | Gradient boosted classifier |
| `flask` | REST API framework (session path only) |
| `joblib==1.3.2` | Model serialization |
| `gunicorn` | WSGI server for session-based serving |
| `requests` | HTTP client for test script |
| `mlflow` | Experiment tracking — pre-installed by CML via `mlflow-cml-plugin`, do not add to `requirements.txt` |
