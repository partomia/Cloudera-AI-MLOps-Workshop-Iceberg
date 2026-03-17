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

### Step 5 — Deploy the Prediction API

#### Option A: Run as a CML Application (recommended)

1. In the project, go to **Applications** > **New Application**.
2. Set the following:
   - **Name**: Credit Risk API
   - **Script**: `03_predict.py`
   - **Kernel**: Python 3
   - **Resource Profile**: 1 vCPU / 2 GB RAM
3. Click **Create Application**.

CML assigns a port via the `CDSW_APP_PORT` environment variable and the app is served by **Gunicorn** (2 workers). Once the status turns green, click the application name or the external link icon to get the public HTTPS endpoint URL.

> **Troubleshooting:** If the application stays on "Starting" or shows `Address already in use`, click the three-dot menu → **Restart**. This clears any stale port binding from a previous crashed instance.

#### Option B: Run interactively in a session

```bash
python 03_predict.py
```

The API starts on `http://localhost:5000` using Gunicorn.

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
