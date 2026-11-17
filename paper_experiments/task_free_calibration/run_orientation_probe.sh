#!/usr/bin/env bash
# Orientation probe: the three phi calibrators, one task per official CALVIN start
# (50 orientation-tolerant + 50 orientation-critical starts, identical across conditions).
REPO=/home/alam/vla_thesis/vla-deployment-benchmark
cd "$REPO"
for cond in body_body world_body body_world; do
  env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL -u _CE_CONDA -u _CE_M \
    /home/alam/miniconda3/bin/python -u paper_experiments/calvin_baselines/run_protocol.py \
    --protocol paper_experiments/task_free_calibration/orient_probe_${cond}.generated.json \
    --workspace /home/alam/vla_thesis --method orient_probe_${cond} --model-seed 0 \
    --num-sequences 1000 --sequence-seed 1729 --skip-memory --overwrite \
    > "$REPO/results/orient_probe_${cond}.out" 2>&1
  echo "[$(date '+%T')] $cond exit $?" >> "$REPO/results/orient_probe.log"
done
echo "[$(date '+%T')] done" >> "$REPO/results/orient_probe.log"
