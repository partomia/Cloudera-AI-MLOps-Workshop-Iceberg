"""
CML Pipeline Orchestrator — triggered by GitHub Actions on push to main.

Runs three CML Jobs in sequence via the CML REST API:
  1. Generate Loan Data       (01_generate_data.py)
  2. Train Credit Risk Model  (02_train_model.py)
  3. Validate Model KPIs      (05_validate_model.py)

Exit codes:
  0 — all jobs succeeded (pipeline green)
  1 — a job failed or a threshold was not met (pipeline red)

Required environment variables (set as GitHub Actions secrets):
  CML_WORKSPACE_URL   e.g. https://ml-xxxx.go01-dem.ylcu-atmi.cloudera.site
  CML_API_KEY         CML API key from your CML profile
  CML_PROJECT_ID      Numeric project ID (visible in CML project URL or Settings)
"""

import os
import sys
import time

import requests

# ── Config ────────────────────────────────────────────────────────────────────
CML_URL    = os.environ["CML_WORKSPACE_URL"].rstrip("/")
API_KEY    = os.environ["CML_API_KEY"]
PROJECT_ID = os.environ["CML_PROJECT_ID"]

HEADERS = {"Authorization": f"Bearer {API_KEY}"}
BASE    = f"{CML_URL}/api/v1/projects/{PROJECT_ID}"

# Must match the Job names created in CML exactly (case-sensitive)
PIPELINE = [
    "Generate Loan Data",
    "Train Credit Risk Model",
    "Validate Model KPIs",
]

POLL_INTERVAL_SEC = 15
JOB_TIMEOUT_SEC   = 900   # 15 minutes per job


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_jobs() -> dict:
    """Return {job_name: job_id} for all jobs in the project."""
    resp = requests.get(f"{BASE}/jobs", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return {j["name"]: j["id"] for j in resp.json().get("jobs", [])}


def trigger_job(job_id: str) -> str:
    """Start a job run and return the run ID."""
    resp = requests.post(
        f"{BASE}/jobs/{job_id}/runs",
        headers=HEADERS,
        json={},
        timeout=30,
    )
    resp.raise_for_status()
    return str(resp.json()["id"])


def wait_for_run(job_id: str, run_id: str) -> bool:
    """Poll until the run finishes. Returns True on success, False otherwise."""
    # CML v1 API status values
    TERMINAL_OK  = {"ENGINE_SUCCEEDED"}
    TERMINAL_BAD = {"ENGINE_FAILED", "ENGINE_STOPPED", "ENGINE_TIMEDOUT"}

    deadline = time.time() + JOB_TIMEOUT_SEC
    while time.time() < deadline:
        resp = requests.get(
            f"{BASE}/jobs/{job_id}/runs/{run_id}",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json().get("status", "")

        if status in TERMINAL_OK:
            return True
        if status in TERMINAL_BAD:
            print(f"  Run ended with status: {status}")
            return False

        time.sleep(POLL_INTERVAL_SEC)

    print(f"  Timed out after {JOB_TIMEOUT_SEC}s waiting for run {run_id}")
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

print(f"CML Pipeline starting — project: {PROJECT_ID}")
print(f"Workspace: {CML_URL}\n")

try:
    available_jobs = list_jobs()
except requests.RequestException as exc:
    print(f"ERROR: Could not reach CML API — {exc}")
    sys.exit(1)

for job_name in PIPELINE:
    if job_name not in available_jobs:
        print(
            f"ERROR: Job '{job_name}' not found in project.\n"
            f"  Create it in CML (Jobs → New Job) with that exact name."
        )
        sys.exit(1)

    job_id = available_jobs[job_name]
    print(f"▶  {job_name}")

    try:
        run_id = trigger_job(job_id)
    except requests.RequestException as exc:
        print(f"  ERROR triggering job: {exc}")
        sys.exit(1)

    print(f"   Run ID : {run_id}")

    ok = wait_for_run(job_id, run_id)
    if not ok:
        print(f"   ✗ FAILED\n\nPipeline aborted at: {job_name}")
        sys.exit(1)

    print(f"   ✓ passed\n")

print("✅  All pipeline stages passed — model is production-ready.")
sys.exit(0)
