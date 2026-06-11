#!/usr/bin/env bash
set -euo pipefail

seeds=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--seeds)
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        seeds+=("$1")
        shift
      done
      ;;
    *)
      shift
      ;;
  esac
done

for seed in "${seeds[@]}"; do
  echo "Seed ${seed}"
  python main.py -c configs.yaml -s "${seed}"
done