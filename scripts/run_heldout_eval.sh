#!/usr/bin/env bash
# Supervised run of the LLM evaluations: red team (held-out + development sets), then answers.
#
#   bash scripts/run_heldout_eval.sh start    # refuses if a run is already active
#   bash scripts/run_heldout_eval.sh status
#   bash scripts/run_heldout_eval.sh stop     # stop flag + kill the whole process tree
#
# Safety rules (added 2026-10-23 after a retry loop outlived a manual stop):
#   - one run at a time: an atomic lock directory .eval_run/lock holds the run's state;
#   - state.json records run id, Windows PIDs, start time, git HEAD and cache sizes (the
#     checkpoint: every LLM call is cached in evals/cache/, so a re-run resumes from there);
#   - retries are bounded (MAX_ATTEMPTS per step) and only for transient network errors;
#     the daily Gemini quota or any other failure stops the run (exit 3 / 2);
#   - `stop` writes a STOP flag (checked before every attempt) and kills the supervisor
#     first, then its children, so nothing can relaunch.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"; RUN_DIR="$ROOT/.eval_run"; LOCK="$RUN_DIR/lock"
PY="${PY:-../farwa_env/Scripts/python.exe}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
STEPS="${STEPS:-redteam answers}"            # overridable only for testing the supervisor
MODULE_PREFIX="${MODULE_PREFIX-src.evals.}"
TRANSIENT='APIConnectionError|ConnectError|ConnectTimeout|ReadTimeout|timed out|RemoteProtocolError|Server disconnected|503|UNAVAILABLE'

winpid() { cat "/proc/$1/winpid" 2>/dev/null || echo "$1"; }
cache_sizes() { for f in evals/cache/*.jsonl; do printf '"%s": %s, ' "$(basename "$f")" "$(grep -c . "$f")"; done; }

status() {
    if [ -d "$LOCK" ]; then cat "$LOCK/state.json"; [ -f "$LOCK/child" ] && echo "current step: $(cat "$LOCK/child")"
        [ -f "$LOCK/STOP" ] && echo "(stop requested)"
    else echo "no active run"; fi
    echo "evaluation processes:"
    wmic process where "name='python.exe'" get ProcessId,CommandLine 2>/dev/null | grep -i "src\.evals" \
        | grep -v wmic || echo "  none"
    [ -n "${MODULE_PREFIX}" ] || wmic process where "name='python.exe'" get ProcessId,CommandLine 2>/dev/null         | grep -i "sleeper" | grep -v wmic
}

stop() {
    [ -d "$LOCK" ] || { echo "no active run"; status; exit 0; }
    touch "$LOCK/STOP"
    sup=$(sed -n 's/.*"supervisor_winpid": \([0-9]*\).*/\1/p' "$LOCK/state.json")
    [ -n "$sup" ] && taskkill //PID "$sup" //T //F >/dev/null 2>&1
    # anything left (e.g. a child that was re-parented)
    for pid in $(wmic process where "name='python.exe'" get ProcessId,CommandLine 2>/dev/null \
                 | grep -i "src\.evals" | grep -v wmic | awk '{print $NF}'); do
        taskkill //PID "$pid" //T //F >/dev/null 2>&1
    done
    sleep 1
    rm -rf "$LOCK"
    echo "stopped"; status
}

start() {
    mkdir -p "$RUN_DIR"
    if ! mkdir "$LOCK" 2>/dev/null; then
        echo "REFUSED: a run is active or was not stopped cleanly:"; status
        echo "If no evaluation process is listed, remove $LOCK and start again."; exit 4
    fi
    run_id="run-$(date +%Y%m%d-%H%M%S)"; log="$RUN_DIR/$run_id.log"
    cat > "$LOCK/state.json" <<EOF
{"run_id": "$run_id", "supervisor_pid": $$, "supervisor_winpid": $(winpid $$),
 "started": "$(date '+%Y-%m-%d %H:%M:%S %z')", "git_head": "$(git rev-parse --short HEAD)",
 "log": "$log", "cache_at_start": {$(cache_sizes)"_": 0}, "max_attempts": $MAX_ATTEMPTS}
EOF
    trap 'rm -rf "$LOCK"' EXIT
    export PYTHONUTF8=1 PYTHONUNBUFFERED=1
    echo "=== $run_id START $(date +%H:%M:%S) HEAD $(git rev-parse --short HEAD) cache {$(cache_sizes)} ===" >> "$log"
    for step in $STEPS; do
        attempt=1
        while :; do
            [ -f "$LOCK/STOP" ] && { echo "=== STOP flag, not starting $step ===" >> "$log"; exit 5; }
            echo "=== $step ATTEMPT $attempt $(date +%H:%M:%S) ===" >> "$log"
            "$PY" -m "$MODULE_PREFIX$step" >> "$log" 2>&1 & child=$!
            echo "$step attempt $attempt, child winpid $(winpid $child)" > "$LOCK/child"
            wait "$child"; rc=$?
            rm -f "$LOCK/child"
            echo "=== $step EXIT $rc $(date +%H:%M:%S) cache {$(cache_sizes)} ===" >> "$log"
            [ $rc -eq 0 ] && break
            [ -f "$LOCK/STOP" ] && exit 5
            if tail -60 "$log" | grep -qE "daily_quota|QuotaExhausted"; then
                echo "=== DAILY QUOTA: stopped; re-run after the reset resumes from the cache ===" >> "$log"; exit 3
            fi
            if ! tail -60 "$log" | grep -qE "$TRANSIENT" || [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
                echo "=== FAILED (not transient, or $MAX_ATTEMPTS attempts used): stopped ===" >> "$log"; exit 2
            fi
            attempt=$((attempt + 1)); sleep 60
        done
    done
    echo "=== $run_id ALL DONE $(date +%H:%M:%S) ===" >> "$log"
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "usage: $0 start|stop|status"; exit 64 ;;
esac
