"""
app/api/test_runner.py — Dashboard-triggered test and experiment runner.

Provides POST /api/tests/run endpoint that:
  1. Runs the full pytest suite programmatically
  2. Runs all 5 comparison experiments
  3. Writes timestamped results CSV
  4. Returns summary JSON to the dashboard
"""

from __future__ import annotations

import csv
import io
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/tests", tags=["tests"])


RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Track the path to the latest CSV result
_latest_csv: str = ""


def _find_python() -> str:
    """Find the Python interpreter."""
    # Try common locations
    import shutil
    for name in ["python3.11", "python3", "python"]:
        path = shutil.which(name)
        if path:
            return path
    return sys.executable


@router.post("/run")
async def run_tests():
    """
    Run the full test suite + 5 experiments.
    Returns combined results as JSON.
    """
    global _latest_csv

    start = time.time()
    python = _find_python()
    project_root = str(Path(__file__).resolve().parent.parent.parent)

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": python,
        "project_root": project_root,
    }

    # ── 1. Run pytest ─────────────────────────────────────────────────────────
    try:
        proc = subprocess.run(
            [python, "-m", "pytest", "tests/", "-v", "--tb=short", "-q"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        test_output = proc.stdout + proc.stderr

        # Parse pass/fail counts from last line
        lines = [l.strip() for l in test_output.strip().split("\n") if l.strip()]
        summary_line = lines[-1] if lines else ""

        passed = failed = errors = 0
        if "passed" in summary_line:
            import re
            m = re.search(r"(\d+) passed", summary_line)
            if m:
                passed = int(m.group(1))
            m = re.search(r"(\d+) failed", summary_line)
            if m:
                failed = int(m.group(1))
            m = re.search(r"(\d+) error", summary_line)
            if m:
                errors = int(m.group(1))

        results["tests"] = {
            "status": "PASSED" if failed == 0 and errors == 0 else "FAILED",
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "summary_line": summary_line,
            "output": test_output[-2000:] if len(test_output) > 2000 else test_output,
        }
    except subprocess.TimeoutExpired:
        results["tests"] = {
            "status": "TIMEOUT",
            "passed": 0, "failed": 0, "errors": 0,
            "summary_line": "Test run timed out after 120s",
            "output": "",
        }
    except Exception as e:
        results["tests"] = {
            "status": "ERROR",
            "passed": 0, "failed": 0, "errors": 1,
            "summary_line": str(e),
            "output": "",
        }

    # ── 2. Run 5 experiments ──────────────────────────────────────────────────
    try:
        # Import and run experiments directly (no subprocess needed)
        sys.path.insert(0, project_root)
        from evaluation.experiments import run_all_experiments, save_results_csv, generate_summary

        exp_results = run_all_experiments(quick=True)
        csv_path = save_results_csv(exp_results, str(RESULTS_DIR))
        _latest_csv = csv_path
        exp_summary = generate_summary(exp_results)

        results["experiments"] = {
            "status": "COMPLETED",
            "total_data_points": len(exp_results),
            "csv_path": csv_path,
            "summary": exp_summary,
        }
    except Exception as e:
        logger.error(f"Experiment run failed: {e}")
        results["experiments"] = {
            "status": "ERROR",
            "error": str(e),
            "total_data_points": 0,
            "csv_path": "",
            "summary": {},
        }

    # ── 3. Write combined CSV ─────────────────────────────────────────────────
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        combined_csv = RESULTS_DIR / f"full_run_{timestamp}.csv"

        with open(combined_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "Item", "Status", "Details"])

            # Tests
            test_data = results.get("tests", {})
            writer.writerow(["Tests", "Suite", test_data.get("status", "N/A"), test_data.get("summary_line", "")])
            writer.writerow(["Tests", "Passed", test_data.get("passed", 0), ""])
            writer.writerow(["Tests", "Failed", test_data.get("failed", 0), ""])

            # Experiments
            exp_data = results.get("experiments", {})
            writer.writerow(["Experiments", "Status", exp_data.get("status", "N/A"), ""])
            writer.writerow(["Experiments", "Data Points", exp_data.get("total_data_points", 0), ""])
            if exp_data.get("csv_path"):
                writer.writerow(["Experiments", "CSV", exp_data["csv_path"], ""])

        results["combined_csv"] = str(combined_csv)
        _latest_csv = str(combined_csv)
    except Exception as e:
        logger.error(f"Combined CSV write failed: {e}")

    elapsed = time.time() - start
    results["elapsed_seconds"] = round(elapsed, 2)

    return results


@router.get("/latest-csv")
async def download_latest_csv():
    """Download the latest test results CSV."""
    if not _latest_csv or not Path(_latest_csv).exists():
        # Try to find the most recent CSV
        csvs = sorted(RESULTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not csvs:
            return {"error": "No CSV results available. Run tests first."}
        return FileResponse(
            str(csvs[0]),
            media_type="text/csv",
            filename=csvs[0].name,
        )
    return FileResponse(
        _latest_csv,
        media_type="text/csv",
        filename=Path(_latest_csv).name,
    )


@router.get("/results")
async def list_results():
    """List all available result files."""
    files = []
    for f in sorted(RESULTS_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        files.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified": time.ctime(f.stat().st_mtime),
        })
    return {"results": files}
