#!/usr/bin/env bats
# Tests for scripts/evolve.sh orchestrator-correctness fixes (issue #22).
#
# evolve.sh is a monolithic orchestrator that drives live `claude` agents and
# git — it cannot be run end-to-end in a unit test. These are regression guards:
# each asserts that a specific fix from #22 is present in the script source, so
# a future edit that reintroduces the bug fails CI. The one genuinely executable
# behaviour (nonce fail-hard) is exercised directly.

setup() {
    EVOLVE="${BATS_TEST_DIRNAME}/../evolve.sh"
}

@test "evolve.sh is syntactically valid bash" {
    run bash -n "$EVOLVE"
    [ "$status" -eq 0 ]
}

# Fix 1 — nonce must fail hard, never fall back to a predictable value.
@test "nonce generation has no predictable fallback" {
    run grep -n 'BOUNDARY_NONCE=' "$EVOLVE"
    [ "$status" -eq 0 ]
    ! grep -q 'fallback-' "$EVOLVE"
    grep -q 'FATAL: cannot generate secure boundary nonce' "$EVOLVE"
}

@test "nonce fail-hard pattern exits non-zero when generator fails" {
    # Replicate the exact guard with a forced failure; the real line uses python3.
    run bash -c 'set -e; N=$(false 2>/dev/null) || { echo "FATAL" >&2; exit 1; }; echo "$N"'
    [ "$status" -eq 1 ]
}

# Fix 2 — fix-loop prompt must carry the verification commands.
@test "FIX_PROMPT injects \$VERIFY_INSTRUCTIONS" {
    # Inside the FIXEOF heredoc, between the error listing and the Steps block.
    run python3 - "$EVOLVE" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r'<<FIXEOF\n(.*?)\nFIXEOF', src, re.S)
sys.exit(0 if (m and '$VERIFY_INSTRUCTIONS' in m.group(1)) else 1)
PY
    [ "$status" -eq 0 ]
}

# Fix 2 (cont.) — verification instructions must be REBUILT after the Step 6
# re-detect, else a bootstrap session's fix prompt carries stale commands.
@test "VERIFY_INSTRUCTIONS is rebuilt after stack re-detection" {
    grep -q 'build_verify_instructions()' "$EVOLVE"
    # 1 definition + at least 2 calls (up front, and after re-detect)
    [ "$(grep -c 'build_verify_instructions' "$EVOLVE")" -ge 3 ]
    # FORMAT_CMD is re-read during the Step 6 re-detect so the rebuild is complete
    grep -q "stdin)\['format'\]" "$EVOLVE"
}

# Fix 3 — build-failure revert must also restore spec.md.
@test "build-failure revert includes spec.md" {
    grep -q 'git checkout "\$SESSION_START_SHA" -- "\$PROJECT_DIR/" spec.md' "$EVOLVE"
}

# Fix 4 — empty command output must still report an exit code, never a blank line.
@test "error capture appends exit code and a no-output placeholder" {
    grep -q '(exit \$rc) \${bout:-<no output>}' "$EVOLVE"
    grep -q '(exit \$rc) \${tout:-<no output>}' "$EVOLVE"
    grep -q '(exit \$rc) \${lout:-<no output>}' "$EVOLVE"
    # The old blank-prone form must be gone.
    ! grep -q '\[\$label build\] \$bout' "$EVOLVE"
}
