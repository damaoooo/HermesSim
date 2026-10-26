#!/bin/bash
# gsat P-Code lifting on Shaheen III, one exclusive node per array task. Each task lifts one LPT-balanced
# shard of the jobs file (lifting/lift_cluster.py run). Resubmit the same command to resume: finished
# outputs are skipped. Per-binary results: <out>/_status/shard_*.jsonl, gsat logs: <out>/_logs/.
#
#   sbatch --array=0-7 scripts/shaheen/lift.sbatch <jobs.jsonl> <out_dir>
#
# Tunables (environment): THREADS (CPU threads per JVM, default 1: more threads barely speed up one JVM), HEAP_GB (minimum -Xmx, default 16),
# ANALYSIS_TIMEOUT (s, default 14400; gsat's own default 1200 cancels the analysis of the largest binaries), DISABLE_ANALYZERS (comma list, default none).
#SBATCH --job-name=hermes-lift
#SBATCH --partition=workq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
USAGE="usage: sbatch --array=0-N scripts/shaheen/lift.sbatch jobs.jsonl out_dir"
JOBS=$(realpath "${1:?$USAGE}")
OUT=$(realpath -m "${2:?$USAGE}")
REPO=${REPO:-$SLURM_SUBMIT_DIR}
cd "$REPO"

# Node-local scratch for Ghidra projects, logs and caches (never Lustre).
LOCAL=/tmp/hermes-lift.${SLURM_JOB_ID}
mkdir -p "$LOCAL"
trap 'rm -rf "$LOCAL"' EXIT
export TMPDIR=$LOCAL
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "node=$(hostname) array=${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1} jobs=$JOBS out=$OUT"
.venv/bin/python lifting/lift_cluster.py run \
    --jobs "$JOBS" --output_dir "$OUT" \
    --jar bin/gsat-1.0-opt.jar --java "$REPO/.jdk/bin/java" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --cores "${SLURM_CPUS_PER_TASK:-192}" --threads "${THREADS:-1}" --heap-gb "${HEAP_GB:-16}" \
    --analysis-timeout "${ANALYSIS_TIMEOUT:-14400}" --disable-analyzers "${DISABLE_ANALYZERS:-}" \
    --local-tmp "$LOCAL"
