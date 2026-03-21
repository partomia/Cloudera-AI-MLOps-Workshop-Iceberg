#!/bin/bash
set -e   # abort on any error

# Step 1: install dependencies
pip install -r requirements.txt

# Step 2: generate synthetic training data
python 01_generate_data.py

# Step 3: train the model and save pkl artifacts into the image
# Both scripts are deterministic (numpy seed 42 / random_state 42).
# Running here ensures credit_risk_model.pkl and label_encoder.pkl are
# baked into the Docker image so cml_model.py can load them at startup.
python 02_train_model.py
