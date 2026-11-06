#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Detect and activate virtualenv if available
if [ -d "$PROJECT_DIR/../.venv" ]; then
    VENV_PATH="$PROJECT_DIR/../.venv"
elif [ -d "$PROJECT_DIR/.venv" ]; then
    VENV_PATH="$PROJECT_DIR/.venv"
fi

if [ -n "$VENV_PATH" ]; then
    # shellcheck disable=SC1091
    source "$VENV_PATH/bin/activate"
fi

# Ensure adk binary is available
if ! command -v adk &> /dev/null; then
    echo "ERROR: 'adk' command not found."
    echo "Please activate your virtual environment or install google-adk:"
    echo "  source /home/udhayan/Agent/.venv/bin/activate"
    exit 1
fi

BRIDGE_HOST="localhost"
BRIDGE_PORT=9875

echo "Checking FreeCAD Robust MCP Bridge at ${BRIDGE_HOST}:${BRIDGE_PORT}..."

# Check bridge connectivity using python socket connection
if ! python3 -c "import socket; s = socket.create_connection(('${BRIDGE_HOST}', ${BRIDGE_PORT}), timeout=2); s.close()" 2>/dev/null; then
    echo ""
    echo "========================================================================="
    echo "❌ ERROR: FreeCAD Robust MCP Bridge is NOT running on ${BRIDGE_HOST}:${BRIDGE_PORT}."
    echo "========================================================================="
    echo "The draftsman agent requires the FreeCAD XML-RPC bridge to execute CAD tools."
    echo ""
    echo "How to start the FreeCAD bridge:"
    echo "  1. Launch FreeCAD with GUI on your desktop."
    echo "  2. In the workbench dropdown, select 'Robust MCP Bridge'."
    echo "  3. Click 'Start MCP Bridge' on the toolbar (or menu)."
    echo "  4. Confirm it reports listening on localhost:${BRIDGE_PORT}."
    echo "  5. Re-run ./run.sh."
    echo "========================================================================="
    echo ""
    exit 1
fi

echo "✅ FreeCAD Robust MCP Bridge is running on ${BRIDGE_HOST}:${BRIDGE_PORT}."
echo "Starting ADK web on port 7020..."
exec adk web --port 7020
