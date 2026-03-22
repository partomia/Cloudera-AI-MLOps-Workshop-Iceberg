"""
cml_pipeline.py — CI/CD pipeline: generate → train → validate
Uses CML REST API v2 directly with requests (no cmlapi dependency needed)
"""
import os
import sys
import time
import requests

# ── Config from GitHub Secrets ────────────────────────────────────────────────
WORKSPACE_URL = os.environ["CML_WORKSPACE_URL"].rstrip("/")
PROJECT_ID    = os.environ["CML_PROJECT_ID"]
API_KEY       = os.environ["CML_API_KEY"]

BASE_URL = f"{WORKSPACE_URL}/api/v2"
HEADERS  = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}

# Runtime config
POLL_INTERVAL_SEC = 15
JOB_TIMEOUT_SEC   = 900   # 15 minutes per job

# Must match CML Job names exactly (case-sensitive)
PIPELINE = [
    "Generate Loan Data",
    "Train Credit Risk Model",
    "Validate Model KPIs",
]

# CML API v2 terminal statuses
TERMINAL_OK  = {"ENGINE_SUCCEEDED"}
TERMINAL_BAD = {"ENGINE_FAILED", "ENGINE_STOPPED", "ENGINE_TIMEDOUT"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_jobs() -> dict:
    """Return {job_name: job_id} for all jobs in the project."""
    resp = requests.get(
        f"{BASE_URL}/projects/{PROJECT_ID}/jobs",
        headers=HEADERS,
        timeout=30,
    )
    resp.raise_for_status()
    return {j["name"]: j["id"] for j in resp.json().get("jobs", [])}


def trigger_job(job_id: str) -> str:
    """Start a job run and return the run ID."""
    resp = requests.post(
        f"{BASE_URL}/projects/{PROJECT_ID}/jobs/{job_id}/runs",
        headers=HEADERS,
        json={},
        timeout=30,
    )
    resp.raise_for_status()
    return str(resp.json()["id"])


def wait_for_run(job_id: str, run_id: str) -> bool:
    """Poll until the run finishes. Returns True on success, False otherwise."""
    deadline = time.time() + JOB_TIMEOUT_SEC
    while time.time() < deadline:
        resp = requests.get(
            f"{BASE_URL}/projects/{PROJECT_ID}/jobs/{job_id}/runs/{run_id}",
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
print(f"Workspace: {WORKSPACE_URL}\n")

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
