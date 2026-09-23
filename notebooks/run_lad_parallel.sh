#!/usr/bin/env bash
# Fan out the LAD-grouped spatial CV: one process per (dim, repeat) pair.
# Each pair owns a distinct checkpoint + model files, so they run collision-free.
# Resumable: re-running skips folds already in the checkpoints.
#
# Usage:   ./run_lad_parallel.sh [MAX_PARALLEL]
#   MAX_PARALLEL defaults to 16 (all 8 dims x 1 repeat at once).
set -euo pipefail
cd "$(dirname "$0")"

PY="../.venv/bin/python"
PAR="${1:-16}"
DIMS=(128 100 64 32 16 8 4 2)
# Override the repeats with the REPEATS env var, e.g. REPEATS="2 3 4 5 6 7 8 9"
read -ra REPEATS <<< "${REPEATS:-0}"
mkdir -p logs_lad

echo "Launching LAD spatial-CV: ${#DIMS[@]} dims x ${#REPEATS[@]} repeats = $(( ${#DIMS[@]} * ${#REPEATS[@]} )) workers, up to $PAR at once"
echo "Logs -> notebooks/logs_lad/dim<D>_rep<R>.log"

for d in "${DIMS[@]}"; do
  for r in "${REPEATS[@]}"; do
    echo "$d $r"
  done
done | xargs -P "$PAR" -I{} bash -c '
  set -- {}; d=$1; r=$2
  echo "[start] dim=$d rep=$r  $(date +%T)"
  if '"$PY"' run_lad_spcv.py --dim "$d" --repeat "$r" > logs_lad/dim${d}_rep${r}.log 2>&1; then
    echo "[done ] dim=$d rep=$r  $(date +%T)"
  else
    echo "[FAIL ] dim=$d rep=$r  $(date +%T)  -> see logs_lad/dim${d}_rep${r}.log"
  fi
'

echo "All workers finished. Now run notebook 2c_lad_blocked_cv cells from the aggregation section down"
echo "(they load the checkpoints in AE_outputs/.../lad_blocked and build the summary + plots)."
