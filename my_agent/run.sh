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

# Load .env if present
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.env"
    set +a
fi

CAD_BACKEND="${CAD_BACKEND:-cadcraft}"
echo "Configured CAD backend: ${CAD_BACKEND}"

if [ "$CAD_BACKEND" = "cadcraft" ]; then
    CADCRAFT_BIN="$(command -v cadcraft-cli || echo "/usr/bin/cadcraft-cli")"
    if [ ! -x "$CADCRAFT_BIN" ]; then
        echo ""
        echo "========================================================================="
        echo "❌ ERROR: CADCraft CLI ('cadcraft-cli') not found or not executable."
        echo "========================================================================="
        echo "Please install CADCraft or ensure 'cadcraft-cli' is in your PATH."
        echo "========================================================================="
        exit 1
    fi
    echo "✅ CADCraft CLI found at ${CADCRAFT_BIN}."
elif [ "$CAD_BACKEND" = "freecad" ]; then
    BRIDGE_HOST="localhost"
    BRIDGE_PORT=9875

    echo "Checking FreeCAD Robust MCP Bridge at ${BRIDGE_HOST}:${BRIDGE_PORT}..."

    # Check bridge connectivity using python socket connection
    if ! python3 -c "import socket; s = socket.create_connection(('${BRIDGE_HOST}', ${BRIDGE_PORT}), timeout=2); s.close()" 2>/dev/null; then
        echo "FreeCAD bridge is not currently running. Launching FreeCAD GUI automatically..."
        setsid freecad </dev/null >/tmp/freecad.log 2>&1 &

        # Wait up to 10 seconds for bridge to start
        CONNECTED=0
        for i in {1..10}; do
            sleep 1
            if python3 -c "import socket; s = socket.create_connection(('${BRIDGE_HOST}', ${BRIDGE_PORT}), timeout=1); s.close()" 2>/dev/null; then
                CONNECTED=1
                break
            fi
            echo "Waiting for FreeCAD bridge to initialize... (${i}s)"
        done

        if [ "$CONNECTED" -eq 0 ]; then
            echo ""
            echo "========================================================================="
            echo "❌ ERROR: FreeCAD Robust MCP Bridge did not start on ${BRIDGE_HOST}:${BRIDGE_PORT}."
            echo "========================================================================="
            echo "How to start the FreeCAD bridge manually:"
            echo "  1. Launch FreeCAD with GUI on your desktop."
            echo "  2. In the workbench dropdown, select 'Robust MCP Bridge'."
            echo "  3. Click 'Start MCP Bridge' on the toolbar (or menu)."
            echo "  4. Confirm it reports listening on localhost:${BRIDGE_PORT}."
            echo "  5. Re-run ./run.sh."
            echo "========================================================================="
            echo ""
            exit 1
        fi
    fi

    echo "✅ FreeCAD Robust MCP Bridge is running on ${BRIDGE_HOST}:${BRIDGE_PORT}."
fi

echo "Starting ADK web on port 7020..."
exec adk web --port 7020
