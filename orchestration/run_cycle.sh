#!/bin/bash
# orchestration/run_cycle.sh — cron entry point for one
# /exp-orchestrate-cycle invocation. Design:
# docs/decisions/hpc-idle-gpu-orchestration.md
#
# Runs headless (`claude -p`), one CLI process per cron fire, all
# pinned to ORCHESTRATOR_SESSION_ID below (fixed UUID) so every
# fire lands in ONE dedicated session instead of spawning a new
# one each time (Tuan's request, 2026-07-11).
#
# Originally used `-c`/--continue ("resume the most recent
# conversation in this directory"), which turned out to be WRONG:
# recency is directory-scoped, not orchestrator-scoped, so any
# actively-open interactive Claude Code session in this same repo
# (e.g. Tuan working in this file right now) is *more recent* than
# the orchestrator's own last fire and would hijack the next `-c`
# resume -- discovered 2026-07-11 while investigating why a rename
# of the interactive session looked like it might affect session
# count; `-c`'s recency-based resolution, not the rename, was the
# real risk. A fixed --session-id/-r pin sidesteps recency
# entirely: it always resumes the exact same session regardless of
# what else is open in this directory.
#
# First-ever fire: the session doesn't exist on disk yet, so it's
# created via --session-id. Every fire after: resumed via -r on
# that same fixed ID. This does NOT change the skill's state
# model: the prompt still tells Claude to re-read queue.yaml/
# jobs.yaml fresh every cycle rather than trust anything
# remembered from a prior turn -- the yaml files remain the only
# source of truth, conversation history is not state.
#
# stdout/stderr go to cron_output.log for debugging cron/claude
# invocation issues themselves (auth, PATH, crashes before the
# skill even loads). This is separate from each launched
# experiment's own log (W&B is the log of record for those, per
# the skill's design) and from orchestration/log.md (the cycle
# log the skill itself appends to on success).
#
# Caution: a continued session accumulates context across all
# fires since the schedule was armed -- watch for auto-compaction
# behavior or degraded quality if the 24h/96-fire window runs
# long.

set -uo pipefail

REPO_DIR="/home/u20/tnguyen9210/tnn1/LLMs/llm-reasoning-methods"
LOG_DIR="$REPO_DIR/orchestration"
LOG_FILE="$LOG_DIR/cron_output.log"
STOP_FILE="$LOG_DIR/cron_stop_at.txt"
# Absolute path -- cron's minimal environment does not source
# .bashrc/.profile, so PATH additions made there (this is where
# `claude` normally resolves from) are invisible to this script.
# Root cause of a ~5h silent outage on 2026-07-11 (all cycles from
# 00:30 to 05:15 failed with "claude: command not found" while
# exiting 0, because `set -e` was never in effect for that
# specific failure to trip on) -- verified this exact path exists
# and matches `which claude` in an interactive shell before fixing.
CLAUDE_BIN="/home/u20/tnguyen9210/.local/bin/claude"
# Fixed UUID for the dedicated orchestrator session -- generated
# once (python3 -c "import uuid; print(uuid.uuid4())") and pinned
# here permanently. Do not regenerate; changing this loses the
# thread and starts a new one on the next fire.
ORCHESTRATOR_SESSION_ID="0c50d77d-78d8-4ed5-add1-cbbb22c86e24"
SESSION_FILE="$HOME/.claude/projects/-home-u20-tnguyen9210-tnn1-LLMs-llm-reasoning-methods/$ORCHESTRATOR_SESSION_ID.jsonl"

# Bound the log file so it can't grow unbounded over a long-lived
# cron schedule (~35 lines/run x 96 runs/day).
if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE")" -gt 20000 ]; then
    tail -n 10000 "$LOG_FILE" > "$LOG_FILE.tmp" \
        && mv "$LOG_FILE.tmp" "$LOG_FILE"
fi

# Self-disable past the stop time written at crontab-install time
# (a fixed timestamp, not "24h from whenever this line first
# runs" -- inspect/extend/cut short by editing this file directly;
# deleting it makes the schedule run indefinitely).
if [ -f "$STOP_FILE" ]; then
    stop_epoch="$(date -d "$(cat "$STOP_FILE")" +%s 2>/dev/null)"
    now_epoch="$(date +%s)"
    if [ -n "$stop_epoch" ] && [ "$now_epoch" -ge "$stop_epoch" ]; then
        {
            echo "===== $(date '+%Y-%m-%d %H:%M:%S') stop time" \
                 "$(cat "$STOP_FILE") reached -- skipping cycle." \
                 "Remove crontab entry or delete $STOP_FILE to" \
                 "continue. ====="
        } >> "$LOG_FILE" 2>&1
        exit 0
    fi
fi

{
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') cron fire ====="
    cd "$REPO_DIR" || exit 1

    if [ ! -x "$CLAUDE_BIN" ]; then
        echo "FAILED: CLAUDE_BIN not executable: $CLAUDE_BIN"
        echo "===== $(date '+%Y-%m-%d %H:%M:%S') cron done, exit=127 ====="
        exit 0
    fi

    if [ -f "$SESSION_FILE" ]; then
        SESSION_FLAGS=(-r "$ORCHESTRATOR_SESSION_ID")
    else
        SESSION_FLAGS=(--session-id "$ORCHESTRATOR_SESSION_ID")
    fi

    "$CLAUDE_BIN" -p "${SESSION_FLAGS[@]}" \
        -n "Run exp-orchestrate-cycle job orchestration" \
        "Run one /exp-orchestrate-cycle invocation now. Follow
.claude/skills/exp-orchestrate-cycle/SKILL.md exactly: refresh
orchestration/jobs.yaml from squeue, read orchestration/queue.yaml
fresh, probe pooled jobs for idle GPUs (0% util AND 0MiB memory),
launch as many planned entries as idle capacity allows respecting
expected_hr walltime guards, mark launched entries running with a
launch: block via targeted edits, append one dated block to
orchestration/log.md, then validate both yaml files parse. This is
one fire of a recurring scheduled job -- even though this may be a
continuation of a prior cycle's conversation thread, treat nothing
remembered from earlier turns as current: re-read both yaml files
and squeue's live output fresh, since either may have changed
since the last fire. Do not ask clarifying questions, just execute
the cycle as specified and stop."
    claude_exit=$?
    if [ "$claude_exit" -ne 0 ]; then
        echo "FAILED: claude -p exited $claude_exit"
    fi
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') cron done, exit=$claude_exit ====="
} >> "$LOG_FILE" 2>&1
