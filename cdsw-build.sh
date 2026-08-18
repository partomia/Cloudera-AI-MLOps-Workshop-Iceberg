#!/bin/bash
set -e

# Install dependencies only.
#
# Do NOT run 01_load_gold.py or 02_train_model.py here. This script executes
# inside the Docker build environment, which has no Data Connection and no
# Kerberos credentials — the Impala read fails and the build errors out.
# The original repo ran both here because 01 generated synthetic data locally.
#
# Model artifacts are produced by the CML Jobs pipeline and read from the
# project filesystem at deployment time:
#   Job 1  01_load_gold.py       -> loan_data.csv
#   Job 2  02_train_model.py     -> credit_risk_model.pkl, label_encoders.pkl
#   Job 3  05_validate_model.py  -> KPI gate

pip install -r requirements.txt