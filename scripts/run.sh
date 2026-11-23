#!/bin/bash
# scripts/run.sh — called by HTCondor, not run directly.
# Replace YOUR_USERNAME with your actual cluster username.

set -e

USERNAME=${USER:-YOUR_USERNAME}
PROJECT_DIR="/nethome/$USERNAME/projects/Transformer"
VENV_DIR="/nethome/$USERNAME/virtualenvs/Transformer"
CONFIG="${1:-configs/sample.yaml}"

echo "================================================"
echo "Job started : $(date)"
echo "Node        : $(hostname)"
echo "Config      : $CONFIG"
echo "================================================"

cd "$PROJECT_DIR"
git pull origin main

source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PROJECT_DIR"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

python main.py --config="$CONFIG"

echo "Job finished: $(date)"
