#!/usr/bin/env bash
# Stage-1 full pipeline: train all six PPO tasks to completion, and record a tracking
# video from the matching *-Play-v0 config right after each run finishes.
#
# All six use the SAME budget (3000 iters / 4096 envs) so the Base vs Oracle vs Waq
# comparison is not confounded by training length.
#
# Usage:
#   ./run_full_pipeline.sh                 # all six
#   TASKS="rough:waq flat:waq" ./run_full_pipeline.sh
#   NUM_ENVS=2048 MAX_ITER=1000 ./run_full_pipeline.sh
#
# wandb logs online (project=lec_dreamwaq). If you are not authenticated,
# run with WANDB_MODE=offline ./run_full_pipeline.sh and sync later.

set -uo pipefail
set -m          # Each background job gets its own process group → on hang, kill the whole group

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
VIDEO_LENGTH="${VIDEO_LENGTH:-400}"     # 400 steps = 1 episode = 20 s
VIDEO_ENVS="${VIDEO_ENVS:-32}"
# rough:terrainoracle = the oracle the paper talks about (height-map policy). Not applicable to flat, which has no terrain info.
TASKS="${TASKS:-rough:basedwq rough:oracledwq rough:waq rough:terrainoracle flat:basedwq flat:oracledwq flat:waq}"
LOG_DIR="${LOG_DIR:-$ROOT/pipeline_logs}"

# Hang watchdog: if the log isn't updated for STALL_SECS, kill it and move on to the next task.
# A normal iteration takes ~10s. Boot + terrain generation can take long, so allow 900s of slack.
# (This watchdog assumes WANDB_CONSOLE=off below — if wandb hijacks stdout, the log
#  stops updating and a healthy training run is misjudged as a hang.)
STALL_SECS="${STALL_SECS:-900}"
CHECK_SECS="${CHECK_SECS:-60}"
# Isaac Sim boot failures (hang / exit 0 after 0 iterations) usually succeed on retry.
TRAIN_TRIES="${TRAIN_TRIES:-3}"

# Isaac Sim / IsaacLab entry points. Override if your install differs.
# Use the bundled kit python (CLAUDE.md rule). Plain python is not on PATH.
PYBIN="${PYBIN:-$HOME/IsaacLab/_isaac_sim/python.sh}"
ISAACLAB_SCRIPTS="${ISAACLAB_SCRIPTS:-$HOME/IsaacLab/scripts/reinforcement_learning/rsl_rl}"

export WANDB_MODE="${WANDB_MODE:-online}"
# wandb online by default hijacks the console (wrap_raw) and diverts stdout into its own output.log.
# Then our training log stops growing, so (a) you can't see progress and
# (b) the stall watchdog above misjudges a healthy run as a hang. Metric logging is unaffected.
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$LOG_DIR"

info() { printf '\033[1;36m[INFO]\033[0m  %s\n' "$*"; }
ok()   { printf '\033[1;32m[ OK ]\033[0m  %s\n' "$*"; }
err()  { printf '\033[1;31m[ERR ]\033[0m  %s\n' "$*"; }

# Stage 1 comparison axis: 3 variants differing only in actor observation, on the official env recipe x 2 terrains.
# (The original DreamWaQ recipe, DreamWaQ-Manager-Go2-*, fails to walk — report.qmd §1)
pkg_dir_for()  { echo dreamwaq_manager; }
task_id_for()  {
    local terrain="$1" variant="$2"          # terrain=flat|rough, variant=basedwq|oracledwq|waq
    local t="${terrain^}"                    # flat -> Flat
    case "$variant" in
        basedwq)   printf 'DreamWaQ-BaseDwq-%s-PPO'   "$t" ;;
        oracledwq) printf 'DreamWaQ-OracleDwq-%s-PPO' "$t" ;;
        waq)       printf 'DreamWaQ-Waq-Official-%s-PPO' "$t" ;;
        terrainoracle) printf 'DreamWaQ-TerrainOracle-%s-PPO' "$t" ;;
        *) return 1 ;;
    esac
}
# experiment_name of the agent cfg — the folder where checkpoints/tfevents/videos accumulate
exp_name_for() {
    local terrain="$1" variant="$2"
    local t="${terrain^}"
    case "$variant" in
        basedwq)   printf 'BaseDwq-Official-%s-PPO-v0'   "$t" ;;
        oracledwq) printf 'OracleDwq-Official-%s-PPO-v0' "$t" ;;
        waq)       printf 'Waq-Official-%s-PPO-v0'       "$t" ;;
        terrainoracle) printf 'TerrainOracle-Official-%s-PPO-v0' "$t" ;;
        *) return 1 ;;
    esac
}

# Run while watching the log file's mtime. If not updated for STALL_SECS, kill the whole process group and return 124.
# (Intermittent native hangs have been reported on the Waq/CENet path, so this is essential for unattended runs.)
run_watched() {
    local logf="$1" pkg="$2"; shift 2
    : > "$logf"
    (
        cd "$ROOT/$pkg/scripts" || exit 127
        export PYTHONPATH="$ISAACLAB_SCRIPTS:${PYTHONPATH:-}"
        exec "$PYBIN" "$@"
    ) > "$logf" 2>&1 &
    local pid=$! age rc
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$CHECK_SECS"
        age=$(( $(date +%s) - $(stat -c %Y "$logf" 2>/dev/null || echo 0) ))
        if (( age > STALL_SECS )); then
            err "  ↳ ${logf##*/}: not updated for ${age}s — treating as hang, kill"
            kill -KILL -- "-$pid" 2>/dev/null
            wait "$pid" 2>/dev/null
            return 124
        fi
    done
    wait "$pid"; rc=$?
    return "$rc"
}

declare -a RESULTS=()

# Also record progress to a file. Even if whoever is watching the console log (human or agent) gets disconnected,
# this file alone tells you which task is running and how far it has gotten.
STATUS_FILE="$LOG_DIR/STATUS.txt"
status_line() { printf '%s  %-22s %s\n' "$(date '+%F %T')" "$1" "$2" >> "$STATUS_FILE"; }

run_one() {
    local stack="$1" variant="$2"
    local pkg base_id train_task play_task
    pkg="$(pkg_dir_for "$stack")"
    base_id="$(task_id_for "$stack" "$variant")" || { err "unknown task: $stack:$variant"; return 1; }
    train_task="${base_id}-v0"
    play_task="${base_id}-Play-v0"
    local tag="${stack}_${variant}"

    info "════════════════════════════════════════"
    info " TRAIN  $train_task   envs=$NUM_ENVS iters=$MAX_ITER"
    info "════════════════════════════════════════"

    # Isaac Sim boot fails intermittently — it either hangs with the log frozen (the watchdog catches
    # this as 124), or exits 0 without running a single iteration (2026-08-30 flat:basedwq).
    # Neither is a config problem; retrying just works. Previously failures were only recorded and we moved
    # on to the next task, so someone had to rerun it by hand later — this removes that manual step.
    local exp run_dir trc attempt why
    exp="$(exp_name_for "$stack" "$variant")"
    for (( attempt = 1; attempt <= TRAIN_TRIES; attempt++ )); do
        (( attempt > 1 )) && info " RETRY  $train_task   attempt $attempt/$TRAIN_TRIES ($why)"
        run_watched "$LOG_DIR/${tag}_train.log" "$pkg" \
            rsl_rl/train.py --task="$train_task" --headless \
                --num_envs="$NUM_ENVS" --seed="$SEED" --max_iterations="$MAX_ITER" \
                --logger=wandb --log_project_name=lec_dreamwaq
        trc=$?

        run_dir="$(ls -1dt "$ROOT/$pkg/logs/rsl_rl/$exp"/*/ 2>/dev/null | head -1)"
        why=""
        if (( trc == 124 )); then
            why="STALLED"
        elif (( trc != 0 )); then
            why="rc=$trc"
        elif [[ -z "$run_dir" ]] \
             || ! compgen -G "$run_dir/events.out.tfevents.*" > /dev/null \
             || ! compgen -G "$run_dir/model_*.pt" > /dev/null; then
            # rc=0 but no outputs. The watchdog only catches 'log stopped' hangs, so it can't catch this.
            why="no outputs"
        else
            break
        fi

        err "$tag: training failed ($why) — see $LOG_DIR/${tag}_train.log"
        # Delete the half-built run folder. If left behind it gets mixed with the folder of the next attempt,
        # and play.py's 'latest run' resolution picks the empty folder.
        if [[ -n "$run_dir" ]] && ! compgen -G "$run_dir/events.out.tfevents.*" > /dev/null; then
            rm -rf "$run_dir"
        fi
        run_dir=""
        (( attempt < TRAIN_TRIES )) && sleep 20
    done

    if [[ -z "$run_dir" ]]; then
        err "$tag: training failed on all ${TRAIN_TRIES} attempts ($why)"
        RESULTS+=( "$tag: TRAIN FAILED x${TRAIN_TRIES} ($why)" )
        status_line "$tag" "TRAIN FAILED x${TRAIN_TRIES} ($why)"
        return 1
    fi
    ok "$tag: training done${attempt:+ (attempt $attempt)}"
    status_line "$tag" "trained (attempt $attempt)"

    info " VIDEO  $play_task   ${VIDEO_LENGTH} steps, tracking env 0"
    # No --load_run/--checkpoint: the cfg defaults (load_run=".*", load_checkpoint="model_.*.pt")
    # resolve to the newest run and newest checkpoint, i.e. the run we just finished.
    # The video step fails intermittently with a Vulkan crash (exit 139) but is unrelated to checkpoints, so
    # retry once; if it still fails, just record it and move on to the next task.
    local vattempt vrc=1
    for vattempt in 1 2; do
        run_watched "$LOG_DIR/${tag}_video.log" "$pkg" \
            rsl_rl/play.py --task="$play_task" --headless \
                --num_envs="$VIDEO_ENVS" --video --video_length="$VIDEO_LENGTH" \
                --track_agent --track_env_index=0
        vrc=$?
        (( vrc == 0 )) && break
        err "$tag: video attempt $vattempt failed (rc=$vrc)"
    done

    # Look directly in the run folder just written (time-window-based search gives false hits on long plays).
    local mp4
    mp4="$(find "${run_dir:-/nonexistent}" -name '*.mp4' 2>/dev/null | sort | tail -1)"

    if (( vrc != 0 )) || [[ -z "$mp4" ]]; then
        err "$tag: training succeeded, no video — see $LOG_DIR/${tag}_video.log"
        RESULTS+=( "$tag: trained, VIDEO FAILED" )
        status_line "$tag" "trained, VIDEO FAILED"
        return 0
    fi
    ok "$tag: video → $mp4"
    RESULTS+=( "$tag: OK  $mp4" )
    status_line "$tag" "OK  $mp4"
    return 0
}

info "pipeline start — tasks: $TASKS"
info "envs=$NUM_ENVS iters=$MAX_ITER seed=$SEED  WANDB_MODE=$WANDB_MODE  STALL=${STALL_SECS}s"
info "checkpoints/tfevents/videos → $ROOT/dreamwaq_manager/logs/rsl_rl/<experiment_name>/<timestamp>/"
info "wandb local copy           → $ROOT/dreamwaq_manager/wandb/"
info "pipeline console logs      → $LOG_DIR/"
info "progress status file       → $STATUS_FILE  (if the session drops, just look here)"
start_ts=$(date +%s)

: > "$STATUS_FILE"
status_line "pipeline" "start — tasks: $TASKS (envs=$NUM_ENVS iters=$MAX_ITER tries=$TRAIN_TRIES)"

for entry in $TASKS; do
    status_line "${entry%%:*}_${entry##*:}" "TRAIN started"
    run_one "${entry%%:*}" "${entry##*:}"
done

status_line "pipeline" "done"

end_ts=$(date +%s)
echo
info "════════════════════════════════════════"
info " pipeline done in $(( (end_ts - start_ts) / 60 )) min"
info "════════════════════════════════════════"
for r in "${RESULTS[@]}"; do
    if [[ "$r" == *OK* ]]; then ok "$r"; else err "$r"; fi
done
info "wandb (${WANDB_MODE}) → project 'lec_dreamwaq'"
info "outputs → $ROOT/dreamwaq_manager/logs/rsl_rl/"
