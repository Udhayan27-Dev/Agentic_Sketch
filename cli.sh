#!/usr/bin/env bash
set -e

PROJECT_DIR="/home/udhayan/Agent/my_agent"
VENV_PYTHON="/home/udhayan/Agent/.venv/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    echo "ERROR: Virtual environment python not found at $VENV_PYTHON"
    exit 1
fi

exec "$VENV_PYTHON" "$PROJECT_DIR/scripts/run_cli.py" "$@"
