#!/usr/bin/env bash
set -euo pipefail
pwd

for seed in 42 1000 12345; do
  echo "Seed ${seed}"
  python main.py -c configs.yaml -s "${seed}"
done