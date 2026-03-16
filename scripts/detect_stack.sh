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

# ── Main detection ──

# Try root directory first
detect_single_stack "$PROJECT_DIR"

# If unknown, scan immediate subdirectories for monorepo layout
if [ "$STACK" = "unknown" ]; then
    SUBSTACKS=""
    SUBSTACK_COUNT=0

    for subdir in "$PROJECT_DIR"/*/; do
        [ -d "$subdir" ] || continue
        detect_single_stack "$subdir"
        if [ "$STACK" != "unknown" ]; then
            SUBDIR_NAME=$(basename "$subdir")
            [ "$SUBSTACK_COUNT" -gt 0 ] && SUBSTACKS="$SUBSTACKS,"
            SUBSTACKS="$SUBSTACKS{\"name\":\"$SUBDIR_NAME\",\"dir\":\"$SUBDIR_NAME\",\"stack\":\"$STACK\",\"build\":\"$BUILD_CMD\",\"test\":\"$TEST_CMD\",\"lint\":\"$LINT_CMD\",\"format\":\"$FORMAT_CMD\"}"
            SUBSTACK_COUNT=$((SUBSTACK_COUNT + 1))
        fi
    done

    if [ "$SUBSTACK_COUNT" -gt 1 ]; then
        # Monorepo: multiple substacks detected
        cat <<EOF
{
  "stack": "monorepo",
  "substacks": [$SUBSTACKS],
  "build": "",
  "test": "",
  "lint": "",
  "format": ""
}
EOF
        exit 0
    elif [ "$SUBSTACK_COUNT" -eq 1 ]; then
        # Single substack found in subdir — re-detect to set variables
        for subdir in "$PROJECT_DIR"/*/; do
            [ -d "$subdir" ] || continue
            detect_single_stack "$subdir"
            if [ "$STACK" != "unknown" ]; then
                break
            fi
        done
    fi
    # If 0 substacks, STACK remains "unknown" — fall through to single output
fi

# Single-stack output (or unknown)
cat <<EOF
{
  "stack": "$STACK",
  "build": "$BUILD_CMD",
  "test": "$TEST_CMD",
  "lint": "$LINT_CMD",
  "format": "$FORMAT_CMD"
}
EOF
