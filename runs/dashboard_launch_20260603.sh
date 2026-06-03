#!/usr/bin/env bash
set -euo pipefail

cd /home/xjw/code/deeplearning/PlanetaryFeatureMatch
export PYTHONPATH=python:scripts

/home/xjw/.local/share/mamba/envs/plascan/bin/python -m pfm_dashboard.app \
  --host 127.0.0.1 \
  --port 7860 \
  --project-root /home/xjw/code/deeplearning/PlanetaryFeatureMatch
