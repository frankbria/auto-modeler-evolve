"""Security + correctness tests for scripts/detect_stack.sh.

These run the real shell script via subprocess (no mocking) and assert that:
  1. Output is always valid JSON.
  2. Normal stacks (python, typescript, monorepo) are detected with the exact
     fixed command strings.
  3. A maliciously named sibling directory cannot inject JSON structure /
     override the fixed build/test/lint/format command strings (the RCE vector
     from issue #2 — basenames were string-interpolated into JSON unescaped). A
     directory that *is* a buildable stack but has an unsafe name makes the
     script fail closed (non-zero exit) rather than silently skip it.

Regression guard for: [P1.2] command injection / RCE in code-evolve orchestrator.
"""

import json
import subprocess
from pathlib import Path

import pytest

# tests -> backend -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "detect_stack.sh"

# Allowed key set for every emitted substack object.
SUBSTACK_KEYS = {"name", "dir", "stack", "build", "test", "lint", "format"}

# Exact command strings the fixed internal map emits for the fixtures below.
EXPECTED_CMDS = {
    "python": {
        "build": "pip install -e .",
        "test": "pytest",
        "lint": "ruff check .",
        "format": "ruff format --check .",
    },
    "typescript": {
        "build": "npm run build",
        "test": "npm run test",
        "lint": "npm run lint",
        "format": "",
    },
}


def run_detect(project_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(SCRIPT), str(project_dir)],
        capture_output=True,
        text=True,
    )


def make_python_pkg(d: Path) -> None:
    # No uv.lock / poetry.lock -> the plain pip command map.
    d.mkdir(parents=True, exist_ok=True)
    (d / "pyproject.toml").write_text("[project]\nname = 'x'\nversion = '0.0.0'\n")


def make_ts_pkg(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "package.json").write_text(
        '{"name":"x","scripts":{"build":"tsc","test":"jest","lint":"eslint"}}'
    )
    (d / "tsconfig.json").write_text("{}")


def assert_exact_cmds(substack: dict) -> None:
    expected = EXPECTED_CMDS[substack["stack"]]
    for field, value in expected.items():
        assert substack[field] == value, (
            f"{substack['name']}.{field} = {substack[field]!r}, expected {value!r}"
        )


def test_single_python_stack_is_valid_json(tmp_path):
    make_python_pkg(tmp_path)
    res = run_detect(tmp_path)
    assert res.returncode == 0
    data = json.loads(res.stdout)  # must be valid JSON
    assert data["stack"] == "python"
    for field, value in EXPECTED_CMDS["python"].items():
        assert data[field] == value


def test_monorepo_detection_uses_exact_fixed_commands(tmp_path):
    make_python_pkg(tmp_path / "backend")
    make_ts_pkg(tmp_path / "frontend")
    res = run_detect(tmp_path)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["stack"] == "monorepo"
    subs = {s["name"]: s for s in data["substacks"]}
    assert set(subs) == {"backend", "frontend"}
    for s in data["substacks"]:
        assert set(s.keys()) == SUBSTACK_KEYS
        assert s["dir"] == s["name"]
        assert_exact_cmds(s)


def test_single_nested_stack_keeps_dir(tmp_path):
    # A lone nested stack must keep its cwd-bearing substack shape (issue #2 /
    # CodeRabbit): promoting it to a top-level single output would drop `dir`
    # and run commands from the project root instead of the subdirectory.
    make_python_pkg(tmp_path / "backend")
    res = run_detect(tmp_path)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["stack"] == "monorepo"
    assert [s["dir"] for s in data["substacks"]] == ["backend"]
    assert_exact_cmds(data["substacks"][0])


def test_malicious_dirname_fails_closed(tmp_path):
    # Legit substacks plus a sibling whose name is a JSON-structure-injection
    # payload trying to smuggle in `"build":"touch HACKED"`. The payload dir is
    # itself a buildable stack, so the script must fail closed.
    make_python_pkg(tmp_path / "backend")
    make_ts_pkg(tmp_path / "frontend")
    evil = tmp_path / 'evil","build":"touch HACKED","x":"'
    make_ts_pkg(evil)

    res = run_detect(tmp_path)
    # Non-zero exit (fail closed) and the payload never reaches any output.
    assert res.returncode != 0
    assert "HACKED" not in res.stdout
    assert "touch HACKED" not in res.stdout
    # Whatever did print on stdout must remain parseable / injection-free.
    if res.stdout.strip():
        json.loads(res.stdout)


@pytest.mark.parametrize(
    "evil_name",
    [
        'a";touch HACKED;"',
        "a`touch HACKED`",
        "a$(touch HACKED)",
        "a b",  # space
        'a","build":"x',
    ],
)
def test_unsafe_named_stack_dirs_fail_closed(tmp_path, evil_name):
    make_python_pkg(tmp_path / "good")
    make_python_pkg(tmp_path / evil_name)  # unsafe name on a real stack
    res = run_detect(tmp_path)
    assert res.returncode != 0
    assert "HACKED" not in res.stdout


def test_unsafe_named_nonstack_dir_is_ignored(tmp_path):
    # An unsafe directory name that is NOT a buildable stack is irrelevant and
    # must not break detection of the legitimate substacks beside it.
    make_python_pkg(tmp_path / "backend")
    make_ts_pkg(tmp_path / "frontend")
    junk = tmp_path / "junk; rm -rf /"
    junk.mkdir(parents=True, exist_ok=True)
    (junk / "README.md").write_text("not a stack")

    res = run_detect(tmp_path)
    assert res.returncode == 0
    data = json.loads(res.stdout)
    names = sorted(s["name"] for s in data["substacks"])
    assert names == ["backend", "frontend"]
