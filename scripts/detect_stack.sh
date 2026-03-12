#!/bin/bash
# scripts/detect_stack.sh — Detect project stack and output build/test/lint commands as JSON.
# Called by evolve.sh to adapt verification to whatever the project is using.
#
# Usage: ./scripts/detect_stack.sh [project_dir]
# Output: JSON with build, test, lint, format commands (or empty strings if not applicable)

set -euo pipefail

PROJECT_DIR="${1:-.}"

# Initialize commands
BUILD_CMD=""
TEST_CMD=""
LINT_CMD=""
FORMAT_CMD=""
STACK="unknown"

# Detection order: most specific first

if [ -f "$PROJECT_DIR/Cargo.toml" ]; then
    STACK="rust"
    BUILD_CMD="cargo build"
    TEST_CMD="cargo test"
    LINT_CMD="cargo clippy --all-targets -- -D warnings"
    FORMAT_CMD="cargo fmt -- --check"

elif [ -f "$PROJECT_DIR/package.json" ]; then
    # Detect package manager
    PKG_MGR="npm"
    if [ -f "$PROJECT_DIR/bun.lockb" ] || [ -f "$PROJECT_DIR/bun.lock" ]; then
        PKG_MGR="bun"
    elif [ -f "$PROJECT_DIR/pnpm-lock.yaml" ]; then
        PKG_MGR="pnpm"
    elif [ -f "$PROJECT_DIR/yarn.lock" ]; then
        PKG_MGR="yarn"
    fi

    # Check for TypeScript
    if [ -f "$PROJECT_DIR/tsconfig.json" ]; then
        STACK="typescript"
        BUILD_CMD="$PKG_MGR run build"
        # Detect test runner
        if grep -q '"vitest"' "$PROJECT_DIR/package.json" 2>/dev/null; then
            TEST_CMD="$PKG_MGR run test"
        elif grep -q '"jest"' "$PROJECT_DIR/package.json" 2>/dev/null; then
            TEST_CMD="$PKG_MGR run test"
        else
            TEST_CMD="$PKG_MGR run test"
        fi
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
    if grep -q '"next"' "$PROJECT_DIR/package.json" 2>/dev/null; then
        STACK="nextjs"
        BUILD_CMD="$PKG_MGR run build"
    fi

elif [ -f "$PROJECT_DIR/pyproject.toml" ]; then
    STACK="python"
    # Check for uv
    if command -v uv &>/dev/null && [ -f "$PROJECT_DIR/uv.lock" ]; then
        BUILD_CMD="uv sync"
        TEST_CMD="uv run pytest"
        LINT_CMD="uv run ruff check ."
        FORMAT_CMD="uv run ruff format --check ."
    elif [ -f "$PROJECT_DIR/poetry.lock" ]; then
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

elif [ -f "$PROJECT_DIR/requirements.txt" ]; then
    STACK="python"
    BUILD_CMD="pip install -r requirements.txt"
    TEST_CMD="pytest"
    LINT_CMD="ruff check ."
    FORMAT_CMD=""

elif [ -f "$PROJECT_DIR/go.mod" ]; then
    STACK="go"
    BUILD_CMD="go build ./..."
    TEST_CMD="go test ./..."
    LINT_CMD="go vet ./..."
    FORMAT_CMD="gofmt -l ."

elif [ -f "$PROJECT_DIR/Makefile" ]; then
    STACK="make"
    BUILD_CMD="make"
    TEST_CMD="make test"
    LINT_CMD=""
    FORMAT_CMD=""
fi

# Verify that build/test scripts actually exist in package.json
if [ -f "$PROJECT_DIR/package.json" ]; then
    if [ -n "$BUILD_CMD" ] && ! grep -q '"build"' "$PROJECT_DIR/package.json" 2>/dev/null; then
        BUILD_CMD=""
    fi
    if [ -n "$TEST_CMD" ] && ! grep -q '"test"' "$PROJECT_DIR/package.json" 2>/dev/null; then
        TEST_CMD=""
    fi
    if [ -n "$LINT_CMD" ] && ! grep -q '"lint"' "$PROJECT_DIR/package.json" 2>/dev/null; then
        LINT_CMD=""
    fi
fi

cat <<EOF
{
  "stack": "$STACK",
  "build": "$BUILD_CMD",
  "test": "$TEST_CMD",
  "lint": "$LINT_CMD",
  "format": "$FORMAT_CMD"
}
EOF
