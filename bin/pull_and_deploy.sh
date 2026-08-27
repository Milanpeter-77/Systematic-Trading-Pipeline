#!/bin/bash
#
# Polls origin/main every 5 minutes (via launchd StartInterval) and fast-
# forwards the local checkout if -- and only if -- a clean fast-forward is
# possible. Never merges, rebases, or force-anything. If a pull actually
# changes code, restarts the alpha-factory LaunchAgent, since a running
# Python process does not notice files changing on disk on its own.
#
# Scheduled by: com.milanpeter.tradingpipeline.pulldeploy.plist (StartInterval=300)

set -uo pipefail

REPO_DIR="/Users/milanpeter/Developer/Systematic-Trading-Pipeline"
LOCK_DIR="/tmp/com.milanpeter.tradingpipeline.gitops.lock"
ALPHA_FACTORY_LABEL="com.milanpeter.tradingpipeline.alphafactory"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cd "$REPO_DIR" || { log "ERROR: cannot cd to $REPO_DIR"; exit 1; }

current_branch=$(git symbolic-ref --short HEAD 2>/dev/null)
if [ "$current_branch" != "main" ]; then
    log "ERROR: expected to be on branch 'main', but HEAD is on '${current_branch:-detached}'. Not touching anything."
    exit 1
fi

# --- acquire lock (mkdir is atomic; shared with run_ingestion_and_publish.sh) ---
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    if [ -d "$LOCK_DIR" ]; then
        lock_age=$(( $(date +%s) - $(stat -f%m "$LOCK_DIR" 2>/dev/null || echo 0) ))
        if [ "$lock_age" -gt 900 ]; then
            log "WARNING: stale lock (${lock_age}s old) -- removing and continuing."
            rmdir "$LOCK_DIR" 2>/dev/null
            mkdir "$LOCK_DIR" 2>/dev/null || { log "Could not acquire lock even after clearing stale one, skipping this run."; exit 0; }
        else
            log "Another git operation is in progress (lock age ${lock_age}s) -- skipping this run."
            exit 0
        fi
    fi
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null' EXIT

# --- fetch ---
if ! git fetch origin main --quiet; then
    log "ERROR: git fetch failed (network/auth issue?). Will retry next cycle."
    exit 1
fi

local_sha=$(git rev-parse main)
remote_sha=$(git rev-parse origin/main)

if [ "$local_sha" = "$remote_sha" ]; then
    log "Up to date at ${local_sha:0:8}."
    exit 0
fi

# --- only fast-forward; never merge/rebase/force ---
if git merge-base --is-ancestor main origin/main; then
    if git pull --ff-only origin main --quiet; then
        new_sha=$(git rev-parse main)
        log "Fast-forwarded ${local_sha:0:8} -> ${new_sha:0:8}. Restarting alpha-factory."
        if launchctl kickstart -k "gui/$(id -u)/${ALPHA_FACTORY_LABEL}"; then
            log "alpha-factory restarted."
        else
            log "ERROR: 'launchctl kickstart' failed for ${ALPHA_FACTORY_LABEL} -- restart it manually."
        fi
    else
        log "ERROR: 'git pull --ff-only' failed unexpectedly after the ancestor check passed. Investigate manually."
        exit 1
    fi
else
    log "ERROR: local main and origin/main have diverged (local has commits origin doesn't). Refusing to touch the working tree. Resolve manually: git log --oneline --graph --all"
    exit 1
fi
