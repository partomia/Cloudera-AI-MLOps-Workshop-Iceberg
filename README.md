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
├── 03_predict.py          # Flask REST API serving predictions
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

CML will clone the repo and automatically run `cdsw-build.sh` to install all dependencies listed in `requirements.txt`.

---

### Step 2 — Open a Session (AI Workbench)

1. Inside the project, click **New Session**.
2. Select the following settings:
   - **Editor**: Workbench (or JupyterLab)
   - **Kernel**: Python 3
   - **Resource Profile**: At least 2 vCPU / 4 GB RAM
3. Click **Start Session**.

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

CML will start the Flask app and provide a public HTTPS endpoint URL.

#### Option B: Run interactively in a session

```bash
python 03_predict.py
```

The API starts on `http://localhost:5000`.

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

| Package | Version | Purpose |
|---|---|---|
| pandas | 2.1.0 | Data manipulation |
| numpy | 1.24.3 | Numerical operations |
| scikit-learn | 1.3.0 | Preprocessing, metrics |
| xgboost | 1.7.6 | Gradient boosted classifier |
| flask | 3.0.0 | REST API server |
| joblib | 1.3.2 | Model serialization |
