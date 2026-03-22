# Credit Risk Model — Loan Default Prediction

An end-to-end machine learning project that predicts the probability of a borrower defaulting on a loan. Built with XGBoost and served as a REST API via Flask, designed to run on **Cloudera AI (CML) / AI Workbench**.

---

## Project Overview

Credit risk scoring is a core use case in financial services. This project demonstrates:

- Synthetic loan dataset generation with realistic risk factors
- Binary classification model training (default vs. no default) using XGBoost
- MLflow experiment tracking integrated with CML's native Experiments UI
- Model serialization and deployment as a REST API
- Automated CI/CD pipeline with KPI validation gates on every push

### Features used for prediction

| Feature | Description |
|---|---|
| `loan_amount` | Requested loan amount (USD) |
| `annual_income` | Applicant's annual income (USD) |
| `credit_score` | FICO-style credit score (300–850) |
| `employment_years` | Years in current employment |
| `debt_to_income` | Ratio of monthly debt payments to gross income |
| `num_credit_lines` | Number of open credit lines |
| `num_delinquencies` | Number of past delinquencies |
| `loan_purpose` | Purpose: home, auto, education, personal, business |

---

## Repository Structure

```
Ravi-ML/
├── requirements.txt                  # Python dependencies
├── cdsw-build.sh                     # CML/CDSW environment bootstrap script
├── 01_generate_data.py               # Generate synthetic loan dataset (loan_data.csv)
├── 02_train_model.py                 # Train XGBoost model, log to MLflow, save artifacts
├── 03_predict.py                     # Flask/Gunicorn REST API — session-based serving
├── 04_test_api.py                    # Test the running Flask API with sample requests
├── 05_validate_model.py              # KPI validation gate — used by CI/CD pipeline
├── cml_model.py                      # CML Model Deployment entry point (production path)
└── .github/workflows/retrain.yml     # GitHub Actions CI/CD workflow
```

---

## MLflow Experiment Tracking

This project uses **MLflow** to log model parameters, KPIs, and artifacts. In Cloudera AI the `MLFLOW_TRACKING_URI` environment variable is automatically set in every session — no configuration needed. When running locally, runs are stored in `./mlruns`.

Each training run (`02_train_model.py`) records:

| What is logged | MLflow key |
|---|---|
| Hyperparameters | `n_estimators`, `max_depth`, `learning_rate`, `eval_metric` |
| Dataset stats | `train_samples`, `test_samples`, `default_rate` |
| **KPI / success metrics** | `roc_auc`, `accuracy` |
| Default class metrics | `precision_default`, `recall_default`, `f1_default` |
| No-default class metrics | `precision_no_default`, `recall_no_default`, `f1_no_default` |
| Model artifact | XGBoost model (with signature + input example) |
| Encoder artifact | `label_encoder.pkl` |

> MLflow is pre-installed in CML sessions via `mlflow-cml-plugin`. It is intentionally excluded from `requirements.txt` to avoid breaking CML's pinned version. When running outside CML (e.g. GitHub Actions), MLflow is skipped automatically.

**To view experiment results in CML:** navigate to **Experiments** in the left sidebar of your project.

---

## Using This Project in Cloudera AI (CML)

### Prerequisites

- Access to a Cloudera AI (CML) workspace
- AI Workbench enabled on your CML cluster
- Git credentials configured in your CML profile

---

### Step 1 — Create a New Project from Git

1. Log in to your **Cloudera AI** workspace.
2. Click **New Project** on the Projects page.
3. Choose **Git** as the project source.
4. Enter the repository URL:
   ```
   https://github.com/partomia/Ravi-ML
   ```
5. Set the project name and click **Create Project**.

---

### Step 2 — Open a Session and Install Dependencies

1. Inside the project, click **New Session**.
2. Select:
   - **Editor**: Workbench (or JupyterLab)
   - **Kernel**: Python 3
   - **Resource Profile**: At least 2 vCPU / 4 GB RAM
3. Click **Start Session**.
4. In the **Terminal** tab, run:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 3 — Generate the Dataset

```bash
python 01_generate_data.py
```

Generates `loan_data.csv` (10,000 synthetic loan records).

Expected output:
```
Dataset generated: 10000 rows, default rate: 28.34%
```

---

### Step 4 — Train the Model

```bash
python 02_train_model.py
```

Trains an XGBoost classifier, logs the run to MLflow, and saves:
- `credit_risk_model.pkl` — trained model
- `label_encoder.pkl` — encoder for `loan_purpose`

Expected output:
```
              precision    recall  f1-score   support
           0       0.xx      0.xx      0.xx      xxxx
           1       0.xx      0.xx      0.xx      xxxx

ROC-AUC: 0.xxxx
MLflow run logged  — run_id: ...
Model saved to credit_risk_model.pkl
```

---

### Step 5 — Start the Prediction API

Run the API server from your **session terminal**:

```bash
python 03_predict.py
```

Gunicorn will start on port `5000`:

```
[INFO] Starting gunicorn 25.1.0
[INFO] Listening at: http://0.0.0.0:5000
[INFO] Booting worker with pid: ...
```

Keep this terminal open — the server must stay running for Step 6.

---

### Step 6 — Test the API

```bash
python 04_test_api.py
```

Expected output:
```
Health check: {'status': 'ok'}

[Low-risk applicant]
  Default probability : 0.0312
  Prediction          : 0
  Risk label          : LOW

[High-risk applicant]
  Default probability : 0.8741
  Prediction          : 1
  Risk label          : HIGH
```

---

### API Reference

#### `POST /predict`

**Request body (JSON):**
```json
{
  "loan_amount": 10000,
  "annual_income": 90000,
  "credit_score": 780,
  "employment_years": 10,
  "debt_to_income": 0.15,
  "num_credit_lines": 5,
  "num_delinquencies": 0,
  "loan_purpose": "home"
}
```

**Response:**
```json
{
  "default_probability": 0.0312,
  "prediction": 0,
  "risk_label": "LOW"
}
```

#### `GET /health`

Returns `{"status": "ok"}` when the API is running.

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

### Step A — Create CML Jobs for the Pipeline

In CML → **Jobs → New Job**, create each job:

| Job name | Script | Dependency |
|---|---|---|
| `Generate Loan Data` | `01_generate_data.py` | — |
| `Train Credit Risk Model` | `02_train_model.py` | Generate Loan Data |
| `Validate Model KPIs` | `05_validate_model.py` | Train Credit Risk Model |

Run **Job 1** then **Job 2** in order. This produces `credit_risk_model.pkl` and `label_encoder.pkl` in the project filesystem and logs the run to the **Experiments** tab.

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

3. Under **Example Input**, paste:
   ```json
   {
     "loan_amount": 10000,
     "annual_income": 90000,
     "credit_score": 780,
     "employment_years": 10,
     "debt_to_income": 0.15,
     "num_credit_lines": 5,
     "num_delinquencies": 0,
     "loan_purpose": "home"
   }
   ```
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
      "loan_amount": 10000,
      "annual_income": 90000,
      "credit_score": 780,
      "employment_years": 10,
      "debt_to_income": 0.15,
      "num_credit_lines": 5,
      "num_delinquencies": 0,
      "loan_purpose": "home"
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
    "risk_label": "LOW"
  }
}
```

### Retraining and Redeployment

1. Re-run **Job 2** (Train Credit Risk Model) — new `.pkl` files are written and a new MLflow run is logged
2. Deployed model → **Builds** → **Rebuild** — CML picks up the new `.pkl` files and redeploys with zero downtime

---

## CI/CD Pipeline (GitHub Actions)

Every push to `main` automatically retrains and validates the model.

```
Push to main
     ↓
GitHub Actions fires (.github/workflows/retrain.yml)
     ↓
Step 1: pip install -r requirements.txt
     ↓
Step 2: python 01_generate_data.py   → loan_data.csv
     ↓
Step 3: python 02_train_model.py     → credit_risk_model.pkl
     ↓
Step 4: python 05_validate_model.py
     ↓
✅ ROC-AUC ≥ 0.75 AND F1 ≥ 0.60 → pipeline green
❌ Either threshold missed        → pipeline red
```

No secrets or external services required — the pipeline runs entirely within GitHub Actions.

### KPI thresholds

Defined at the top of `05_validate_model.py`:

```python
ROC_AUC_MIN    = 0.72
F1_DEFAULT_MIN = 0.50
```

These are calibrated to realistic performance on the synthetic dataset. Raise them when switching to real loan data.

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
