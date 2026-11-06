# sketchcad

**sketchcad** is a Google ADK (Agent Development Kit) application that converts an uploaded 2D technical drawing (image or PDF) into a live, parametric drawing inside FreeCAD via MCP (Model Context Protocol).

---

## Architecture Overview

sketchcad implements a sequential 2-agent pipeline:

1. **Analyst (`analyst`)**:
   - Multimodal vision agent powered by `gemini-3.5-flash`.
   - Tool-free agent with a structured `output_schema` (`DrawingSpec`).
   - Executes a 3-pass extraction: Transcribe labels/dimensions $\to$ Structure geometry into mm (origin at bottom-left) $\to$ Consistency check.
   - Saves the extracted specification into session state under `drawing_spec`.

2. **Draftsman (`draftsman`)**:
   - CAD execution agent powered by `gemini-3.5-flash`.
   - Reads `{drawing_spec}` from session state.
   - Interacts with FreeCAD through `freecad-robust-mcp` (`freecad-mcp` in XML-RPC mode on `localhost:9875`).
   - Uses a filtered set of 15 tools: creates document, PartDesign body, XY sketch, draws entities, applies geometric/dimensional constraints, verifies solver status, captures screenshots, and exports DXF.

---

## Prerequisites

1. **Linux OS** with FreeCAD installed locally with GUI.
2. **Robust MCP Bridge Workbench** installed in FreeCAD (`spkane/freecad-addon-robust-mcp-server`).
3. **Python 3.10+** (tested on Python 3.14).
4. Google Gemini API Key.

---

## Installation & Setup

1. **Activate the Virtual Environment**:
   ```bash
   source /home/udhayan/Agent/.venv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` (or update existing `.env`):
   ```bash
   cp .env.example .env
   ```
   Ensure `.env` contains:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   SKETCHCAD_MODEL=gemini-3.5-flash
   ```

---

## Starting the FreeCAD Bridge

Before running the agent, FreeCAD's XML-RPC bridge must be listening on `localhost:9875`:

1. Launch FreeCAD on your desktop.
2. In the workbench selector dropdown, choose **Robust MCP Bridge**.
3. In the toolbar or workbench menu, click **Start MCP Bridge**.
4. FreeCAD's report view will confirm the bridge is listening on `localhost:9875`.

*(Note: Do not expose or rebind ports 9875 or 9876 to external interfaces. The bridge runs unauthenticated Python commands and must remain on localhost).*

---

## Running the Application

### Option A: Using the Launcher (Recommended)
Run the pre-flight verification script, which validates bridge reachability before launching ADK:
```bash
./run.sh
```

### Option B: Direct ADK CLI
```bash
adk web --port 7020
```

---

## Testing & Verification

1. **Validate Agent Import**:
   ```bash
   python -c "import sketchcad_agent.agent; print('Agent imported successfully!')"
   ```

2. **Run Schema Unit Tests**:
   ```bash
   pytest tests/test_schema.py
   ```

3. **Generate Sample CAD Test Drawing**:
   ```bash
   python scripts/make_test_image.py
   ```
   This generates `test_l_shape.png` (an L-shape: 24 mm wide, 12 mm tall, 6 mm wall, with dimension callouts).

---

## End-to-End Workflow

1. Open your browser to `http://localhost:7020`.
2. Select the `sketchcad_agent` from the agent list.
3. In the chat interface, upload `test_l_shape.png`.
4. Send a prompt such as:
   > "Please analyze this 2D drawing and draw it parametrically in FreeCAD."
5. Watch FreeCAD:
   - The **Analyst** produces the `DrawingSpec`.
   - The **Draftsman** creates a new document, body, and sketch, adds the lines, applies constraints, and solves the sketch.
   - The agent outputs the solver status, screenshot, and exported DXF location.
