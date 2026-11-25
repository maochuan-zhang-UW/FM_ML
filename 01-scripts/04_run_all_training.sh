#!/bin/bash
# All training runs behind GJI Figs 4-7 and S3 (45 runs), as executed on 2026-11-22.
# Each line runs 01-scripts/04_train.py
#   <strategy> <training set> <seed> [held-out station for LOSO]
# and writes 06-models/retrain_20261122/<set>_<strategy>_s<seed>[_loso<STA>].{keras,json}.
# Training sets are built with 01-scripts/03_build_datasets.py.
# Jobs run NPAR at a time (default 4); finished runs (JSON present) are skipped.
#
# Usage (from the repository root):  bash 01-scripts/04_run_all_training.sh
set -u
OUT=06-models/retrain_20261122
PY=${PY:-python}
NPAR=${NPAR:-4}
mkdir -p "$OUT"
jobs_list() {
cat <<'JOBS'
scratch STEP 0
freezecls STEP 0
scratch STEP010 0
freezecls STEP010 0
scratch TMSF_Tra_002s 0
freezecls TMSF_Tra_002s 0
scratch TMSF_Tra_000 0 AS1
scratch TMSF_Tra_000 0 AS2
scratch TMSF_Tra_000 0 CC1
scratch TMSF_Tra_000 0 EC1
scratch TMSF_Tra_000 0 EC2
scratch TMSF_Tra_000 0 EC3
scratch TMSF_Tra_000 0 ID1
scratch TMSF_Tra_000 0
freezecls TMSF_Tra_000 0
finetune TMSF_Tra_000 0
probe TMSF_Tra_000 0
scratch TMSF_Tra_001 0
freezecls TMSF_Tra_001 0
finetune TMSF_Tra_001 0
probe TMSF_Tra_001 0
scratch TMSF_Tra_000 1
freezecls TMSF_Tra_000 1
finetune TMSF_Tra_000 1
probe TMSF_Tra_000 1
scratch TMSF_Tra_001 1
freezecls TMSF_Tra_001 1
scratch STEP 1
freezecls STEP 1
scratch STEP010 1
freezecls STEP010 1
scratch TMSF_Tra_002s 1
freezecls TMSF_Tra_002s 1
scratch TMSF_Tra_000 2
freezecls TMSF_Tra_000 2
finetune TMSF_Tra_000 2
probe TMSF_Tra_000 2
scratch TMSF_Tra_001 2
freezecls TMSF_Tra_001 2
scratch STEP 2
freezecls STEP 2
scratch STEP010 2
freezecls STEP010 2
scratch TMSF_Tra_002s 2
freezecls TMSF_Tra_002s 2
JOBS
}
run() {
  set -- $1
  tag="$2_$1_s$3${4:+_loso$4}"
  [ -f "$OUT/$tag.json" ] && { echo "skip $tag"; return; }
  "$PY" 01-scripts/04_train.py "$@" > "$OUT/$tag.log" 2>&1
  echo "finished $tag"
}
export -f run; export OUT PY
jobs_list | xargs -P "$NPAR" -I{} bash -c 'run "{}"'
