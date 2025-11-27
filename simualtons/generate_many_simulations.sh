#!/usr/bin/env bash
set -euo pipefail

# Batch generator for our simulations using simulations.py.
# We generate:
#   - samples S in {5, 10, 15}
#   - clusters K in {20, 25, 30}  (simulations.py uses tree-size = K + 1)
#   - repetitions R in {1..5}
#
# For each (K, S, R) we:
#   - derive a deterministic seed from (K, S, R, BASE_SEED)
#   - call simulations.py
#   - write outputs under: our_simulations/K{K}_S{S}_R{R}
#   - record the seed and noise settings in seed.json in that directory.

SCRIPT="simulations.py"

BASE_SEED=7
MAX_ROOT_OUT_DEG=1
MUTS_PER_CLUSTER=5
FREQ_NOISE_PERCENT=0
CORRUPT_FREQ_PERCENT=0
SIMULATE_READS=1
MUTS_PER_CLUSTER_MIN=20
MUTS_PER_CLUSTER_MAX=50
MEAN_COVERAGE=500
COVERAGE_SHAPE=100
VAF_HETEROZYGOSITY=0.5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-seed|--seed)
      BASE_SEED="$2"; shift 2 ;;
    --mutations-per-cluster|-m)
      MUTS_PER_CLUSTER="$2"; shift 2 ;;
    --max-root-out-deg)
      MAX_ROOT_OUT_DEG="$2"; shift 2 ;;
    --freq-noise-percent|-p)
      FREQ_NOISE_PERCENT="$2"; shift 2 ;;
    --corrupt-freq-percent|-c)
      CORRUPT_FREQ_PERCENT="$2"; shift 2 ;;
    --simulate-reads)
      SIMULATE_READS=1; shift ;;
    --no-simulate-reads)
      SIMULATE_READS=0; shift ;;
    --mutations-per-cluster-min)
      MUTS_PER_CLUSTER_MIN="$2"; shift 2 ;;
    --mutations-per-cluster-max)
      MUTS_PER_CLUSTER_MAX="$2"; shift 2 ;;
    --mean-coverage)
      MEAN_COVERAGE="$2"; shift 2 ;;
    --coverage-shape)
      COVERAGE_SHAPE="$2"; shift 2 ;;
    --vaf-heterozygosity)
      VAF_HETEROZYGOSITY="$2"; shift 2 ;;
    --outdir)
      OUT_ROOT="$2"; shift 2 ;;
    --script)
      SCRIPT="$2"; shift 2 ;;
    --)
      shift; break ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Supported: --base-seed|--seed, --mutations-per-cluster|-m, --max-root-out-deg, --freq-noise-percent|-p, --corrupt-freq-percent|-c, --simulate-reads, --no-simulate-reads, --mutations-per-cluster-min, --mutations-per-cluster-max, --mean-coverage, --coverage-shape, --vaf-heterozygosity, --outdir, --script" >&2
      exit 1 ;;
  esac
done

mkdir -p "$OUT_ROOT"

Ks=({3..20})
Ss=(5 10 15)
Reps=(1 2 3 4 5)

for K in "${Ks[@]}"; do
  for S in "${Ss[@]}"; do
    for R in "${Reps[@]}"; do
      # Derive a deterministic seed per (K,S,R)
      RUN_SEED=$(( BASE_SEED + K * 10000 + S * 100 + R ))

      RUN_NAME=$(printf "K%02d_S%02d_R%02d" "$K" "$S" "$R")
      RUN_DIR="${OUT_ROOT}/${RUN_NAME}"
      mkdir -p "$RUN_DIR"

      TREE_SIZE=$((K + 1))
      echo ">>> Running ${RUN_NAME} with tree-size=${TREE_SIZE}, samples=${S}, seed=${RUN_SEED}"
      cmd=(
        python3 "$SCRIPT"
        --tree-size "$TREE_SIZE"
        --num-samples "$S"
        --seed "$RUN_SEED"
        --max-root-out-deg "$MAX_ROOT_OUT_DEG"
        --mutations-per-cluster "$MUTS_PER_CLUSTER"
        --freq-noise-percent "$FREQ_NOISE_PERCENT"
        --corrupt-freq-percent "$CORRUPT_FREQ_PERCENT"
        --outdir "$RUN_DIR"
        --run-name "$RUN_NAME"
      )
      if [[ "$SIMULATE_READS" -eq 1 ]]; then
        cmd+=(
          --simulate-reads
          --mutations-per-cluster-min "$MUTS_PER_CLUSTER_MIN"
          --mutations-per-cluster-max "$MUTS_PER_CLUSTER_MAX"
          --mean-coverage "$MEAN_COVERAGE"
          --coverage-shape "$COVERAGE_SHAPE"
          --vaf-heterozygosity "$VAF_HETEROZYGOSITY"
        )
      fi
      "${cmd[@]}"

      printf '{ "K": %d, "S": %d, "R": %d, "seed": %d, "freq_noise_percent": %s, "corrupt_freq_percent": %s }\n' \
        "$K" "$S" "$R" "$RUN_SEED" "$FREQ_NOISE_PERCENT" "$CORRUPT_FREQ_PERCENT" > "${RUN_DIR}/seed.json"
    done
  done
done

echo "Done. Outputs in: ${OUT_ROOT}/"
