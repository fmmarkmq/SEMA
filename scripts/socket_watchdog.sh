#!/bin/bash
# Watchdog: SIGINT the GRPO trainer ranks when in-use TCP socket count
# climbs above a threshold, to force a graceful save before ephemeral
# port exhaustion kills the IDA judge calls.
#
# Background: litellm + aiohttp inside sema/utils.get_completions leaks
# TCP sockets at ~20/min. The kernel's ephemeral port range is
# 32768-60999 (28232 ports). Once the in-use count crosses ~28000 the
# gateway becomes unreachable, IDA returns all zeros, and the policy
# collapses (see progress_qwen3b_repro.md, 2026-05-08 entry). We can't
# patch the leak without modifying source code (Rule 5), so this
# watchdog catches the runaway and saves work first.
#
# Usage:
#   bash scripts/socket_watchdog.sh &
# It runs until killed; tail it via tail -F files/logs/socket_watchdog.log.

THRESHOLD="${SOCKET_WATCHDOG_THRESHOLD:-25000}"   # in-use socket count to trip on
POLL_INTERVAL="${SOCKET_WATCHDOG_POLL:-60}"        # seconds between checks
LOG_FILE="files/logs/socket_watchdog.log"
mkdir -p files/logs

log() {
    echo "[$(date -Is)] $*" | tee -a "$LOG_FILE" >&2
}

log "Started. threshold=$THRESHOLD interval=${POLL_INTERVAL}s"

# Tolerate trainer not being up yet at watchdog launch — it may take 1-2 minutes
# for vLLM-serve healthcheck + accelerate init + DeepSpeed init to spawn the
# train_cli.py rl_grpo processes. Only exit on missing procs after we have
# previously seen them.
SEEN_TRAINER=0
MISSING_STREAK=0

while true; do
    INUSE=$(awk '/^TCP/ {print $3}' /proc/net/sockstat)
    PIDS=$(pgrep -f "train_cli.py rl_grpo" | tr '\n' ' ')

    if [ -z "$PIDS" ]; then
        if [ "$SEEN_TRAINER" -eq 0 ]; then
            log "No trainer ranks yet (still starting up)..."
            sleep "$POLL_INTERVAL"
            continue
        fi
        # Trainer used to exist but is gone -> tolerate brief restart windows.
        MISSING_STREAK=$((MISSING_STREAK + 1))
        if [ "$MISSING_STREAK" -ge 3 ]; then
            log "Trainer ranks gone for 3 polls (~$((MISSING_STREAK * POLL_INTERVAL))s); exiting watchdog."
            exit 0
        fi
        log "Trainer ranks missing (streak=$MISSING_STREAK)..."
        sleep "$POLL_INTERVAL"
        continue
    fi
    SEEN_TRAINER=1
    MISSING_STREAK=0

    if [ "$INUSE" -gt "$THRESHOLD" ]; then
        log "TRIPPED: in-use sockets=$INUSE > threshold=$THRESHOLD; sending SIGINT to ranks: $PIDS"
        kill -INT $PIDS
        log "SIGINT sent. Waiting up to 120s for graceful save before exiting watchdog."
        for i in $(seq 1 24); do
            sleep 5
            ALIVE=$(pgrep -f "train_cli.py rl_grpo" | wc -l)
            log "  [$((i*5))s] train_cli ranks alive=$ALIVE"
            if [ "$ALIVE" -eq 0 ]; then
                log "All trainer ranks gone; exiting watchdog."
                exit 0
            fi
        done
        log "Trainer still alive after 120s; force-killing."
        pgrep -f "train_cli.py rl_grpo" | xargs -r kill -9
        exit 0
    else
        log "ok: in-use sockets=$INUSE (threshold=$THRESHOLD)"
    fi

    sleep "$POLL_INTERVAL"
done
