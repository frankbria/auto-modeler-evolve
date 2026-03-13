#!/bin/bash
# scripts/run_evolve_cron.sh — Cron wrapper for evolve.sh
# Sets up the environment that cron doesn't provide (NVM, PATH, etc.)
# Includes lock file to prevent overlapping runs.

set -euo pipefail

LOCKFILE="/tmp/automodeler-evolve.lock"
LOGDIR="/home/frankbria/projects/auto-modeler-evolve/logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/evolve-$(date +%Y-%m-%d_%H-%M).log"

# Prevent overlapping runs
if [ -f "$LOCKFILE" ]; then
    LOCK_PID=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "=== Skipped: previous run (PID $LOCK_PID) still active ===" >> "$LOGFILE"
        exit 0
    fi
    rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT

# Source NVM so `node` and `claude` are available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# Ensure claude is on PATH
export PATH="$HOME/.local/bin:$PATH"

# Load API key from environment file
[ -f "$HOME/.env.automodeler" ] && . "$HOME/.env.automodeler"

# Project directory
cd /home/frankbria/projects/auto-modeler-evolve

# Prevent nesting guard if somehow CLAUDECODE is set
unset CLAUDECODE 2>/dev/null || true

echo "=== Evolution cron started at $(date) ===" >> "$LOGFILE"
./scripts/evolve.sh >> "$LOGFILE" 2>&1
EXIT_CODE=$?
echo "=== Evolution cron finished at $(date) with exit code $EXIT_CODE ===" >> "$LOGFILE"

# Keep only last 30 logs
ls -t "$LOGDIR"/evolve-*.log 2>/dev/null | tail -n +31 | xargs rm -f 2>/dev/null || true
