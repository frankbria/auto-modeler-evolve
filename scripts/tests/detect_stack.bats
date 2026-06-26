#!/usr/bin/env bats
# Tests for scripts/detect_stack.sh — stack detection + command tuples.
# Run: bats scripts/tests/detect_stack.bats   (needs bats-core + python3)

setup() {
    DETECT="${BATS_TEST_DIRNAME}/../detect_stack.sh"
    WORK="$(mktemp -d)"
}

teardown() {
    rm -rf "$WORK"
}

# field <json> <key> — extract a top-level string field
field() {
    python3 -c "import sys,json; print(json.load(sys.stdin)['$2'])" <<<"$1"
}

@test "go: format command writes in place (gofmt -w, not -l)" {
    printf 'module x\n\ngo 1.22\n' > "$WORK/go.mod"
    run bash "$DETECT" "$WORK"
    [ "$status" -eq 0 ]
    [ "$(field "$output" stack)" = "go" ]
    [ "$(field "$output" format)" = "gofmt -w ." ]
}

@test "rust: cargo tuple" {
    printf '[package]\nname="x"\n' > "$WORK/Cargo.toml"
    run bash "$DETECT" "$WORK"
    [ "$(field "$output" stack)" = "rust" ]
    [ "$(field "$output" build)" = "cargo build" ]
}

@test "python+uv: uv run pytest" {
    : > "$WORK/pyproject.toml"
    : > "$WORK/uv.lock"
    run bash "$DETECT" "$WORK"
    [ "$(field "$output" stack)" = "python" ]
    # uv may or may not be installed in CI; build is uv sync only when uv present.
    [[ "$(field "$output" test)" == *pytest* ]]
}

@test "node/yarn: yarn package manager picked from lockfile" {
    printf '{"scripts":{"build":"x","test":"x","lint":"x"}}\n' > "$WORK/package.json"
    : > "$WORK/yarn.lock"
    run bash "$DETECT" "$WORK"
    [ "$(field "$output" build)" = "yarn run build" ]
}

@test "nextjs: detected from package.json dependency" {
    printf '{"dependencies":{"next":"16"},"scripts":{"build":"next build"}}\n' > "$WORK/package.json"
    : > "$WORK/tsconfig.json"
    run bash "$DETECT" "$WORK"
    [ "$(field "$output" stack)" = "nextjs" ]
}

@test "unknown: empty dir yields unknown stack, empty commands" {
    run bash "$DETECT" "$WORK"
    [ "$status" -eq 0 ]
    [ "$(field "$output" stack)" = "unknown" ]
    [ "$(field "$output" build)" = "" ]
}

@test "unsafe substack directory name fails closed" {
    bad="$WORK/a;b"
    mkdir -p "$bad"
    printf '[package]\nname="x"\n' > "$bad/Cargo.toml"
    run bash "$DETECT" "$WORK"
    [ "$status" -ne 0 ]
}
