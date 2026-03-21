# Credit Risk Model — Loan Default Prediction

A end-to-end machine learning project that predicts the probability of a borrower defaulting on a loan. Built with XGBoost and served as a REST API via Flask, designed to run on **Cloudera AI (CML) / AI Workbench**.

---

## Project Overview

Credit risk scoring is a core use case in financial services. This project demonstrates:

- Synthetic loan dataset generation with realistic risk factors
- Binary classification model training (default vs. no default) using XGBoost
- Model serialization and deployment as a REST API
- API validation with sample low-risk and high-risk applicants

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
├── requirements.txt       # Python dependencies
├── cdsw-build.sh          # CML/CDSW environment bootstrap script
├── 01_generate_data.py    # Generate synthetic loan dataset (loan_data.csv)
├── 02_train_model.py      # Train XGBoost model, evaluate, save model artifacts
├── 03_predict.py          # Flask/Gunicorn REST API serving predictions
└── 04_test_api.py         # Test the running API with sample requests
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

**To view experiment results in CML:** navigate to **Experiments** in the left sidebar of your project. Each run appears with all logged metrics, making it easy to compare hyperparameter sweeps or track model improvement over time.

---

## Using This Project in Cloudera AI (CML) — AI Workbench

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
5. Set the project name (e.g., `credit-risk-model`) and click **Create Project**.

CML will clone the repo. The `cdsw-build.sh` script is used to build a custom engine image — it does **not** run automatically inside a session.

---

### Step 2 — Open a Session and Install Dependencies

1. Inside the project, click **New Session**.
2. Select the following settings:
   - **Editor**: Workbench (or JupyterLab)
   - **Kernel**: Python 3
   - **Resource Profile**: At least 2 vCPU / 4 GB RAM
3. Click **Start Session**.
4. Once the session is ready, open the **Terminal** tab and run:
   ```bash
   pip install -r requirements.txt
   ```
   This installs xgboost, flask, joblib, gunicorn, and other dependencies not included in the CML base image.

> **Note:** If your admin has configured a custom engine image using `cdsw-build.sh`, packages will already be pre-installed and you can skip step 4.

---

### Step 3 — Generate the Dataset

Open a terminal or run the script directly in the session:

```bash
python 01_generate_data.py
```

This generates `loan_data.csv` (10,000 synthetic loan records) in the project directory.

Expected output:
```
Dataset generated: 10000 rows, default rate: 28.34%
```

---

### Step 4 — Train the Model

```bash
python 02_train_model.py
```

This reads `loan_data.csv`, trains an XGBoost classifier, prints evaluation metrics, and saves:
- `credit_risk_model.pkl` — trained model
- `label_encoder.pkl` — encoder for `loan_purpose`

Expected output:
```
              precision    recall  f1-score   support
           0       0.xx      0.xx      0.xx      xxxx
           1       0.xx      0.xx      0.xx      xxxx

ROC-AUC: 0.xxxx
Model saved to credit_risk_model.pkl
```

---

### Step 5 — Start the Prediction API

Run the API server from your **session terminal**:

```bash
python 03_predict.py
```

Gunicorn will start on port `5000` and print:

```
=== CML PORT DIAGNOSTICS ===
  CDSW_APP_PORT = ...
  Binding on port: 5000
============================
[INFO] Starting gunicorn 25.1.0
[INFO] Listening at: http://0.0.0.0:5000
[INFO] Booting worker with pid: ...
```

Keep this terminal open (the server must stay running for Step 6).

#### CML Application deployment — known port conflict

The CML **Application** feature assigns ports via `CDSW_APP_PORT`. In some CML deployments this is set to the same value as `CDSW_READONLY_PORT` and `CDSW_PUBLIC_PORT` (e.g. all `8100`), a port CML pre-binds for its own infrastructure. The application can never bind to it, and the Application stays stuck on "Starting".

| Variable | Observed value | Meaning |
|---|---|---|
| `CDSW_APP_PORT` | `8100` | Port CML expects the app to use |
| `CDSW_READONLY_PORT` | `8100` | Pre-bound by CML — unavailable |
| `CDSW_PUBLIC_PORT` | `8100` | External-facing port CML proxies |

The script detects this conflict and falls back to port `5000`, but CML's proxy still points at `8100` so the Application URL won't reach the API. **Resolving this requires a CML admin** to either fix the port assignment or expose the Application on a different port. Until then, use the session-based approach above.

---

### Step 6 — Test the API

Update `BASE_URL` in `04_test_api.py` if using the CML Application endpoint (Option A above), then run:

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

## Dependencies

| Package | Purpose |
|---|---|
| pandas | Data manipulation |
| numpy | Numerical operations |
| scikit-learn | Preprocessing, metrics |
| xgboost | Gradient boosted classifier |
| flask | REST API framework |
| joblib | Model serialization |
| gunicorn | Production WSGI server for CML Applications |
