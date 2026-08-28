#!/bin/bash
#
# Runs once per hour (via launchd StartCalendarInterval, Minute=0):
#   1. Fast-forwards local main from origin first (safety net in case the
#      5-min poller missed a cycle -- never merges/rebases/forces).
#   2. Runs one data-ingestion pass against IBKR TWS. Failure here does NOT
#      block step 4 -- alpha-factory's independent output should still be
#      published even if this hour's ingestion attempt failed.
#   3. Rotates/prunes local log files.
#   4. Commits and pushes any changes under results/ and logs/ so the
#      existing hourly GitHub Actions dashboard workflow has fresh input,
#      and pipeline logs are reviewable without Screen Sharing into this Mac.
#
# Scheduled by: com.milanpeter.tradingpipeline.ingestionpublish.plist

set -uo pipefail

REPO_DIR="/Users/milanpeter/Developer/Systematic-Trading-Pipeline"
VENV_PY="$REPO_DIR/.venv/bin/python"
LOCK_DIR="/tmp/com.milanpeter.tradingpipeline.gitops.lock"
LOG_DIR="/Users/milanpeter/Library/Logs/SystematicTradingPipeline"
MAX_LOG_BYTES=$((5 * 1024 * 1024))  # 5 MB

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "$REPO_DIR" || { log "ERROR: cannot cd to $REPO_DIR"; exit 1; }

current_branch=$(git symbolic-ref --short HEAD 2>/dev/null)
if [ "$current_branch" != "main" ]; then
    log "ERROR: expected to be on branch 'main', but HEAD is on '${current_branch:-detached}'. Aborting."
    exit 1
fi

# --- acquire lock (shared with pull_and_deploy.sh) ---
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    if [ -d "$LOCK_DIR" ]; then
        lock_age=$(( $(date +%s) - $(stat -f%m "$LOCK_DIR" 2>/dev/null || echo 0) ))
        if [ "$lock_age" -gt 900 ]; then
            log "WARNING: stale lock (${lock_age}s old) -- removing and continuing."
            rmdir "$LOCK_DIR" 2>/dev/null
            mkdir "$LOCK_DIR" 2>/dev/null || { log "Could not acquire lock, skipping this run."; exit 0; }
        else
            log "Another git operation is in progress (lock age ${lock_age}s) -- skipping this run."
            exit 0
        fi
    fi
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

# --- 1. fast-forward from origin first, so this run's results commit lands on top of the latest code ---
git_ok=true
if git fetch origin main --quiet; then
    local_sha=$(git rev-parse main)
    remote_sha=$(git rev-parse origin/main)
    if [ "$local_sha" != "$remote_sha" ]; then
        if git merge-base --is-ancestor main origin/main; then
            git pull --ff-only origin main --quiet
            log "Fast-forwarded to $(git rev-parse --short main) before this run."
        else
            log "ERROR: local main and origin/main have diverged. Not touching the working tree. Will skip publishing this run until this is resolved manually (git log --oneline --graph --all)."
            git_ok=false
        fi
    fi
else
    log "WARNING: git fetch failed; proceeding with ingestion using whatever is on disk, but will skip publishing this run."
    git_ok=false
fi

# --- 2. run one ingestion pass (never blocks the publish step below) ---
ingestion_status=0
"$VENV_PY" scripts/run_data_ingestion.py || ingestion_status=$?
if [ "$ingestion_status" -eq 0 ]; then
    log "Ingestion pass completed successfully."
else
    log "WARNING: ingestion pass exited with status $ingestion_status (is TWS/IB Gateway running and logged in? see logs/data_ingestion/ for detail). Continuing to publish step regardless -- alpha-factory's output should still go out."
fi

# --- 3. log housekeeping ---
rotate_if_large() {
    local file="$1"
    if [ -f "$file" ]; then
        local size
        size=$(stat -f%z "$file" 2>/dev/null || echo 0)
        if [ "$size" -gt "$MAX_LOG_BYTES" ]; then
            cp "$file" "${file%.log}_$(date +%Y%m%d_%H%M%S).log.old"
            : > "$file"
            log "Rotated $(basename "$file") (was ${size} bytes)."
        fi
    fi
}
rotate_if_large "$LOG_DIR/pulldeploy.log"
rotate_if_large "$LOG_DIR/ingestion_publish.log"
rotate_if_large "$LOG_DIR/alphafactory.log"
find "$LOG_DIR" -name '*.log.old' -mtime +14 -delete 2>/dev/null
find "$REPO_DIR/logs/data_ingestion" -name '*.log' -mtime +14 -delete 2>/dev/null
find "$REPO_DIR/logs/alpha_factory" -name '*.log' -mtime +14 -delete 2>/dev/null

# --- 4. publish results/ and logs/ if anything changed ---
if [ "$git_ok" = true ]; then
    git add results/ logs/
    if git diff --cached --quiet; then
        log "No changes under results/ to publish."
    else
        git commit --quiet -m "Automated results sync $(date '+%Y-%m-%d %H:%M:%S %Z')"
        if git push origin main --quiet; then
            log "Pushed $(git rev-parse --short main) to origin/main."
        else
            log "ERROR: git push failed (remote moved again? network/auth issue?). The commit is safely stored locally and will be included in the next successful push. Investigate if this repeats."
        fi
    fi
else
    log "Skipping publish step this run due to the git issue above."
fi
