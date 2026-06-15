#!/bin/bash
# scripts/detect_stack.sh — Detect project stack and output build/test/lint commands as JSON.
# Called by evolve.sh to adapt verification to whatever the project is using.
#
# Usage: ./scripts/detect_stack.sh [project_dir]
# Output: JSON with build, test, lint, format commands (or empty strings if not applicable)
#
# Supports monorepos: if no stack marker at root, scans immediate subdirectories.
# When 2+ substacks found, outputs {"stack":"monorepo","substacks":[...]}.

set -euo pipefail

PROJECT_DIR="${1:-.}"

# ── detect_single_stack(dir) ──
# Sets: STACK, BUILD_CMD, TEST_CMD, LINT_CMD, FORMAT_CMD
detect_single_stack() {
    local dir="$1"
    STACK="unknown"
    BUILD_CMD=""
    TEST_CMD=""
    LINT_CMD=""
    FORMAT_CMD=""

    if [ -f "$dir/Cargo.toml" ]; then
        STACK="rust"
        BUILD_CMD="cargo build"
        TEST_CMD="cargo test"
        LINT_CMD="cargo clippy --all-targets -- -D warnings"
        FORMAT_CMD="cargo fmt -- --check"

    elif [ -f "$dir/package.json" ]; then
        # Detect package manager
        local PKG_MGR="npm"
        if [ -f "$dir/bun.lockb" ] || [ -f "$dir/bun.lock" ]; then
            PKG_MGR="bun"
        elif [ -f "$dir/pnpm-lock.yaml" ]; then
            PKG_MGR="pnpm"
        elif [ -f "$dir/yarn.lock" ]; then
            PKG_MGR="yarn"
        fi

        # Check for TypeScript
        if [ -f "$dir/tsconfig.json" ]; then
            STACK="typescript"
            BUILD_CMD="$PKG_MGR run build"
            TEST_CMD="$PKG_MGR run test"
            LINT_CMD="$PKG_MGR run lint"
            FORMAT_CMD=""
        else
            STACK="javascript"
            BUILD_CMD="$PKG_MGR run build"
            TEST_CMD="$PKG_MGR run test"
            LINT_CMD="$PKG_MGR run lint"
            FORMAT_CMD=""
        fi

        # Check for Next.js specifically
        if grep -q '"next"' "$dir/package.json" 2>/dev/null; then
            STACK="nextjs"
            BUILD_CMD="$PKG_MGR run build"
        fi

        # Verify that build/test/lint scripts actually exist in package.json
        if [ -n "$BUILD_CMD" ] && ! grep -q '"build"' "$dir/package.json" 2>/dev/null; then
            BUILD_CMD=""
        fi
        if [ -n "$TEST_CMD" ] && ! grep -q '"test"' "$dir/package.json" 2>/dev/null; then
            TEST_CMD=""
        fi
        if [ -n "$LINT_CMD" ] && ! grep -q '"lint"' "$dir/package.json" 2>/dev/null; then
            LINT_CMD=""
        fi

    elif [ -f "$dir/pyproject.toml" ]; then
        STACK="python"
        if command -v uv &>/dev/null && [ -f "$dir/uv.lock" ]; then
            BUILD_CMD="uv sync"
            TEST_CMD="uv run pytest"
            LINT_CMD="uv run ruff check ."
            FORMAT_CMD="uv run ruff format --check ."
        elif [ -f "$dir/poetry.lock" ]; then
            BUILD_CMD="poetry install"
            TEST_CMD="poetry run pytest"
            LINT_CMD="poetry run ruff check ."
            FORMAT_CMD="poetry run ruff format --check ."
        else
            BUILD_CMD="pip install -e ."
            TEST_CMD="pytest"
            LINT_CMD="ruff check ."
            FORMAT_CMD="ruff format --check ."
        fi

    elif [ -f "$dir/requirements.txt" ]; then
        STACK="python"
        BUILD_CMD="pip install -r requirements.txt"
        TEST_CMD="pytest"
        LINT_CMD="ruff check ."
        FORMAT_CMD=""

    elif [ -f "$dir/go.mod" ]; then
        STACK="go"
        BUILD_CMD="go build ./..."
        TEST_CMD="go test ./..."
        LINT_CMD="go vet ./..."
        FORMAT_CMD="gofmt -l ."

    elif [ -f "$dir/Makefile" ]; then
        STACK="make"
        BUILD_CMD="make"
        TEST_CMD="make test"
        LINT_CMD=""
        FORMAT_CMD=""
    fi
}

# ── JSON emission (injection-safe) ──
# All output is assembled by python3's json.dumps over argv values — NEVER by
# string-interpolating untrusted basenames into a JSON template. Args:
#   $1 stack  $2 build  $3 test  $4 lint  $5 format
#   then flat groups of 6 per substack: name stack build test lint format
# Security note (issue #2): the build/test/lint/format strings come from the
# fixed internal map above; the only attacker-controlled value is a substack
# basename, which is validated against [A-Za-z0-9._-] before it ever reaches here.
emit_json() {
    python3 - "$@" <<'PY'
import json
import sys

argv = sys.argv[1:]
stack, build, test, lint, fmt = argv[:5]
rest = argv[5:]

out = {"stack": stack, "build": build, "test": test, "lint": lint, "format": fmt}

subs = []
for i in range(0, len(rest), 6):
    name, sstack, sbuild, stest, slint, sfmt = rest[i : i + 6]
    subs.append(
        {
            "name": name,
            "dir": name,
            "stack": sstack,
            "build": sbuild,
            "test": stest,
            "lint": slint,
            "format": sfmt,
        }
    )
if subs:
    out["substacks"] = subs

print(json.dumps(out, indent=2))
PY
}

# ── Main detection ──

# Try root directory first
detect_single_stack "$PROJECT_DIR"

# If unknown, scan immediate subdirectories for monorepo layout
if [ "$STACK" = "unknown" ]; then
    SUBARGS=()        # flat 6-tuples (name stack build test lint format) per substack
    SUBSTACK_COUNT=0

    for subdir in "$PROJECT_DIR"/*/; do
        [ -d "$subdir" ] || continue
        detect_single_stack "$subdir"
        # A directory that is not a buildable stack is irrelevant — ignore it
        # (including oddly-named scratch dirs).
        if [ "$STACK" = "unknown" ]; then
            continue
        fi
        SUBDIR_NAME=$(basename "$subdir")
        # Fail closed: a directory that IS a buildable substack but whose name
        # could break JSON/shell quoting is treated as tampering, not silently
        # skipped — otherwise a lone unsafe stack would surface as "unknown" and
        # callers would skip verification of a suspicious layout (issue #2). Such
        # a name can never reach JSON emission.
        if ! [[ "$SUBDIR_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
            printf 'detect_stack: refusing unsafe substack directory name: %q\n' "$SUBDIR_NAME" >&2
            exit 1
        fi
        SUBARGS+=("$SUBDIR_NAME" "$STACK" "$BUILD_CMD" "$TEST_CMD" "$LINT_CMD" "$FORMAT_CMD")
        SUBSTACK_COUNT=$((SUBSTACK_COUNT + 1))
    done

    if [ "$SUBSTACK_COUNT" -ge 1 ]; then
        # One or more nested stacks: always emit cwd-bearing substack metadata
        # (each carries its own "dir") so callers run commands from the correct
        # subdirectory. Even a single nested stack uses this shape — promoting it
        # to a top-level single output would drop the dir and run from the root.
        emit_json "monorepo" "" "" "" "" "${SUBARGS[@]}"
        exit 0
    fi
    # If 0 substacks, STACK remains "unknown" — fall through to single output
fi

# Single-stack output (or unknown)
emit_json "$STACK" "$BUILD_CMD" "$TEST_CMD" "$LINT_CMD" "$FORMAT_CMD"
