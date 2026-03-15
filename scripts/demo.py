#!/usr/bin/env python3
"""
AutoModeler Self-Demo Script
=============================
Exercises the full platform workflow autonomously:
  1. Start backend server (or connect to existing)
  2. Create a project
  3. Load sample data
  4. Run natural-language data query
  5. Get feature suggestions
  6. Set target variable
  7. Train models
  8. Get model comparison
  9. Validate best model
 10. Deploy model
 11. Make a prediction
 12. Download batch predictions
 13. Report results + timing for each step

Run with:
  cd src/backend && uv run python ../../scripts/demo.py
  OR: python scripts/demo.py --url http://localhost:8000

Exit code 0 = all steps passed; non-zero = something failed.
"""

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HTTP helpers (pure stdlib — no requests dependency)
# ---------------------------------------------------------------------------

def _request(method: str, url: str, data: Any = None, headers: dict | None = None, timeout: int = 120) -> tuple[int, Any]:
    """Make an HTTP request; return (status_code, parsed_response)."""
    req_headers = headers or {}

    if isinstance(data, bytes):
        body = data
    elif data is not None:
        body = json.dumps(data).encode()
        req_headers.setdefault("Content-Type", "application/json")
    else:
        body = None

    req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw.decode(errors="replace")


def _get(url: str, **kwargs) -> tuple[int, Any]:
    return _request("GET", url, **kwargs)


def _post(url: str, data: Any = None, **kwargs) -> tuple[int, Any]:
    return _request("POST", url, data=data, **kwargs)


def _delete(url: str) -> tuple[int, Any]:
    return _request("DELETE", url)


def _post_form(url: str, fields: dict, file_field: str, filename: str, file_content: bytes) -> tuple[int, Any]:
    """Multipart form-data POST for file uploads."""
    boundary = "----AutoModelerDemoBoundary"
    body_parts = []

    for name, value in fields.items():
        body_parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )

    body_parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\nContent-Type: text/csv\r\n\r\n'.encode()
        + file_content
        + b'\r\n'
    )
    body_parts.append(f'--{boundary}--\r\n'.encode())

    body = b''.join(body_parts)
    return _request(
        "POST",
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

class DemoRunner:
    def __init__(self, base_url: str, verbose: bool = True):
        self.base = base_url.rstrip("/")
        self.verbose = verbose
        self.results: list[dict] = []
        self.project_id: str | None = None
        self.dataset_id: str | None = None
        self.feature_set_id: str | None = None
        self.best_run_id: str | None = None
        self.deployment_id: str | None = None

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def step(self, name: str, fn):
        """Run a demo step, record timing and pass/fail."""
        self.log(f"\n{'─'*60}")
        self.log(f"  STEP: {name}")
        self.log(f"{'─'*60}")
        start = time.time()
        try:
            result = fn()
            elapsed = time.time() - start
            self.results.append({"step": name, "status": "PASS", "elapsed_s": round(elapsed, 2)})
            self.log(f"  ✓  PASS ({elapsed:.2f}s)")
            return result
        except AssertionError as e:
            elapsed = time.time() - start
            self.results.append({"step": name, "status": "FAIL", "elapsed_s": round(elapsed, 2), "error": str(e)})
            self.log(f"  ✗  FAIL: {e}")
            return None
        except Exception as e:
            elapsed = time.time() - start
            self.results.append({"step": name, "status": "ERROR", "elapsed_s": round(elapsed, 2), "error": str(e)})
            self.log(f"  ✗  ERROR: {e}")
            return None

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def check_health(self):
        status, body = _get(f"{self.base}/health")
        assert status == 200, f"Health check failed: {status}"
        self.log(f"     Server: {self.base}")

    def create_project(self):
        status, body = _post(f"{self.base}/api/projects", {"name": "Demo Project", "description": "Automated demo run"})
        assert status == 201, f"Create project failed: {status} — {body}"
        self.project_id = body["id"]
        self.log(f"     Project ID: {self.project_id}")

    def load_sample_data(self):
        assert self.project_id, "No project_id"
        status, body = _post(f"{self.base}/api/data/sample", {"project_id": self.project_id})
        assert status == 201, f"Load sample failed: {status} — {body}"
        self.dataset_id = body["dataset_id"]
        rows = body["row_count"]
        cols = body["column_count"]
        self.log(f"     Dataset: {body['filename']} — {rows} rows × {cols} cols")
        assert rows > 0, "Dataset has 0 rows"
        assert cols > 0, "Dataset has 0 columns"
        assert len(body.get("preview", [])) > 0, "Preview is empty"
        assert len(body.get("column_stats", [])) > 0, "Column stats missing"

    def query_data(self):
        """Natural language query — requires Anthropic auth token. Soft failure if unavailable."""
        assert self.dataset_id, "No dataset_id"
        status, body = _post(
            f"{self.base}/api/data/{self.dataset_id}/query",
            {"question": "What is the total revenue by region?"},
        )
        assert status == 200, f"Query failed: {status} — {body}"
        # When no API key is available, the endpoint returns a graceful fallback message
        answer = body.get("answer", "")
        assert answer, "Query returned no answer"
        self.log(f"     Answer: {answer[:120]}...")
        if body.get("chart_spec"):
            self.log(f"     Chart: {body['chart_spec']['chart_type']} — {body['chart_spec']['title']}")

    def get_feature_suggestions(self):
        assert self.dataset_id, "No dataset_id"
        status, body = _get(f"{self.base}/api/features/{self.dataset_id}/suggestions")
        assert status == 200, f"Suggestions failed: {status} — {body}"
        suggestions = body.get("suggestions", [])
        self.log(f"     Got {len(suggestions)} feature suggestions")
        if suggestions:
            self.log(f"     First: {suggestions[0]['title']}")

    def apply_features(self):
        """Apply an empty (pass-through) transformation set to create an active FeatureSet.

        This is required before training — training looks for an active FeatureSet
        to know which columns and target variable to use.
        """
        assert self.dataset_id, "No dataset_id"
        # Apply with an empty transformations list — this creates the FeatureSet record
        # with all original columns intact. Users can add transformations interactively.
        status, body = _post(
            f"{self.base}/api/features/{self.dataset_id}/apply",
            {"transformations": []},
        )
        assert status == 201, f"Apply features failed: {status} — {body}"
        self.feature_set_id = body["feature_set_id"]
        self.log(f"     FeatureSet ID: {self.feature_set_id} ({body['total_columns']} columns)")

    def set_target_variable(self):
        assert self.dataset_id, "No dataset_id"
        # Sample data has 'revenue' as target
        status, body = _post(
            f"{self.base}/api/features/{self.dataset_id}/target",
            {"target_column": "revenue", "feature_set_id": self.feature_set_id},
        )
        assert status == 200, f"Set target failed: {status} — {body}"
        self.log(f"     Target: {body['target_column']} — problem type: {body['problem_type']}")

    def train_models(self):
        assert self.project_id, "No project_id"
        status, body = _post(
            f"{self.base}/api/models/{self.project_id}/train",
            {"algorithms": ["linear_regression", "random_forest_regressor"]},
        )
        assert status == 202, f"Train failed: {status} — {body}"
        self.log(f"     Training started — algorithms: {body.get('algorithms', [])}")

        # Poll until done (max 120s)
        deadline = time.time() + 120
        while time.time() < deadline:
            status, runs_body = _get(f"{self.base}/api/models/{self.project_id}/runs")
            assert status == 200
            runs = runs_body.get("runs", [])
            done = [r for r in runs if r["status"] == "done"]
            pending = [r for r in runs if r["status"] in ("pending", "training")]
            if not pending:
                break
            time.sleep(2)

        assert done, "No training runs completed"
        self.log(f"     Completed {len(done)}/{len(runs)} model runs")
        for run in done:
            metrics = run.get("metrics") or {}
            self.log(f"       {run['algorithm']}: {json.dumps(metrics)[:80]}")

    def compare_models(self):
        assert self.project_id, "No project_id"
        status, body = _get(f"{self.base}/api/models/{self.project_id}/compare")
        assert status == 200, f"Compare failed: {status} — {body}"
        models = body.get("models", [])
        rec = body.get("recommendation")
        self.log(f"     {len(models)} models compared")
        if rec:
            self.log(f"     Recommended: {rec['algorithm']} — {rec['reason'][:80]}")
            self.best_run_id = rec["model_run_id"]
        else:
            # Fall back to first done model
            done = [m for m in models if m["status"] == "done"]
            if done:
                self.best_run_id = done[0]["id"]
        assert self.best_run_id, "No best model identified"

    def validate_model(self):
        assert self.best_run_id, "No best_run_id"
        status, body = _get(f"{self.base}/api/validate/{self.best_run_id}/metrics")
        assert status == 200, f"Validation failed: {status} — {body}"
        cv = body.get("cross_validation", {})
        conf = body.get("confidence", {})
        self.log(f"     CV: {cv.get('summary', '')[:80]}")
        self.log(f"     Confidence: {conf.get('overall_confidence')} — {conf.get('summary', '')[:80]}")

    def get_feature_importance(self):
        assert self.best_run_id, "No best_run_id"
        status, body = _get(f"{self.base}/api/validate/{self.best_run_id}/explain")
        assert status == 200, f"Explain failed: {status} — {body}"
        features = body.get("feature_importance", [])
        top3 = features[:3]
        self.log(f"     Top features: {[f['feature'] for f in top3]}")
        self.log(f"     Summary: {body.get('summary', '')[:80]}")

    def deploy_model(self):
        assert self.best_run_id, "No best_run_id"
        status, body = _post(f"{self.base}/api/deploy/{self.best_run_id}")
        assert status in (200, 201), f"Deploy failed: {status} — {body}"
        self.deployment_id = body["id"]
        self.log(f"     Deployment ID: {self.deployment_id}")
        self.log(f"     Dashboard URL: {body.get('dashboard_url', '')}")
        self.log(f"     Endpoint: {body.get('endpoint_path', '')}")

    def make_prediction(self):
        assert self.deployment_id, "No deployment_id"
        # Get feature schema first
        status, dep = _get(f"{self.base}/api/deploy/{self.deployment_id}")
        assert status == 200, f"Get deployment failed: {status}"

        schema = dep.get("feature_schema", [])
        input_data: dict = {}
        for feat in schema:
            if feat["type"] == "numeric":
                input_data[feat["name"]] = feat.get("median", 0) or 50.0
            else:
                opts = feat.get("options", [])
                input_data[feat["name"]] = opts[0] if opts else "unknown"

        status, body = _post(f"{self.base}/api/predict/{self.deployment_id}", input_data)
        assert status == 200, f"Prediction failed: {status} — {body}"
        self.log(f"     Prediction: {body.get('prediction')} ({body.get('target_column', '')})")

    def batch_predict(self):
        assert self.deployment_id, "No deployment_id"
        # Get feature schema to build a mini CSV
        status, dep = _get(f"{self.base}/api/deploy/{self.deployment_id}")
        assert status == 200
        schema = dep.get("feature_schema", [])
        if not schema:
            self.log("     [skipped — no feature schema available]")
            return

        # Build a tiny 3-row CSV
        headers = [f["name"] for f in schema]
        rows = []
        for _ in range(3):
            row = []
            for feat in schema:
                if feat["type"] == "numeric":
                    row.append(str(feat.get("median", 50) or 50))
                else:
                    opts = feat.get("options", [])
                    row.append(opts[0] if opts else "val")
            rows.append(row)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        writer.writerows(rows)
        csv_bytes = buf.getvalue().encode()

        # Upload as multipart form
        boundary = "----DemoBatchBoundary"
        body_bytes = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="batch.csv"\r\nContent-Type: text/csv\r\n\r\n'.encode()
            + csv_bytes
            + f'\r\n--{boundary}--\r\n'.encode()
        )
        req = urllib.request.Request(
            f"{self.base}/api/predict/{self.deployment_id}/batch",
            data=body_bytes,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200, f"Batch predict status {resp.status}"
            result_csv = resp.read().decode()
            result_rows = result_csv.strip().split("\n")
            # +1 for header row
            assert len(result_rows) >= 4, f"Expected 4 rows (header + 3), got {len(result_rows)}"
            self.log(f"     Batch: {len(result_rows)-1} predictions returned")

    def undeploy_cleanup(self):
        assert self.deployment_id, "No deployment_id"
        status, _ = _delete(f"{self.base}/api/deploy/{self.deployment_id}")
        assert status == 204, f"Undeploy failed: {status}"
        self.log(f"     Deployment {self.deployment_id} undeployed")
        # Also delete the project
        if self.project_id:
            _delete(f"{self.base}/api/projects/{self.project_id}")
            self.log(f"     Project {self.project_id} deleted")

    # ------------------------------------------------------------------
    # Run all steps
    # ------------------------------------------------------------------

    def run(self) -> bool:
        steps = [
            ("Health check", self.check_health),
            ("Create project", self.create_project),
            ("Load sample data", self.load_sample_data),
            ("Natural language data query", self.query_data),
            ("Get feature suggestions", self.get_feature_suggestions),
            ("Apply feature transforms", self.apply_features),
            ("Set target variable", self.set_target_variable),
            ("Train models", self.train_models),
            ("Compare models", self.compare_models),
            ("Validate best model", self.validate_model),
            ("Feature importance (SHAP)", self.get_feature_importance),
            ("Deploy model", self.deploy_model),
            ("Single prediction", self.make_prediction),
            ("Batch prediction (CSV)", self.batch_predict),
            ("Undeploy & cleanup", self.undeploy_cleanup),
        ]

        for name, fn in steps:
            self.step(name, fn)

        return self._print_summary()

    def _print_summary(self) -> bool:
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] != "PASS")
        total_time = sum(r["elapsed_s"] for r in self.results)

        print(f"\n{'═'*60}")
        print(f"  AutoModeler Demo Results")
        print(f"{'═'*60}")
        print(f"  {'Step':<40} {'Status':>6}  {'Time':>6}")
        print(f"  {'─'*40} {'──────':>6}  {'──────':>6}")
        for r in self.results:
            icon = "✓" if r["status"] == "PASS" else "✗"
            print(f"  {icon} {r['step']:<38} {r['status']:>6}  {r['elapsed_s']:>5.2f}s")
            if r.get("error"):
                print(f"      Error: {r['error'][:80]}")
        print(f"{'─'*60}")
        print(f"  {passed}/{passed+failed} steps passed  |  Total: {total_time:.1f}s")
        print(f"{'═'*60}")

        return failed == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def wait_for_server(url: str, max_wait: int = 30) -> bool:
    """Poll health endpoint until server is up."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            status, _ = _get(f"{url}/health")
            if status == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser(description="AutoModeler self-demo script")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend base URL")
    parser.add_argument("--wait", type=int, default=0, help="Seconds to wait for server startup")
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step output")
    args = parser.parse_args()

    if args.wait > 0:
        print(f"Waiting up to {args.wait}s for server at {args.url}...")
        if not wait_for_server(args.url, args.wait):
            print(f"ERROR: Server did not become healthy within {args.wait}s")
            sys.exit(2)

    runner = DemoRunner(args.url, verbose=not args.quiet)
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
