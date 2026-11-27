import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional, Union, Any, Dict, List
import math
from dotenv import load_dotenv
from pydantic import BaseModel
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.tools import exit_loop
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from sketchcad_agent.geometry_tools import (
    fillet_two_lines,
    tangent_lines_two_circles,
    tangent_from_point_to_circle,
    circle_circle_intersections,
    circle_line_intersections,
    arc_angles,
    line_line_intersection,
    trim_extend_lines,
    chamfer_two_lines,
    close_joint,
    trim_circle_or_arc,
    tangent_arc_line_circle,
    tangent_arc_two_circles,
    tangent_point_and_vector,
)
from sketchcad_agent.cadcraft_examples import CADCRAFT_FEW_SHOT_EXEMPLARS

load_dotenv()

OPENAI_KEY = os.getenv('OPENAI_API_KEY') or 'EMPTY'
OPENAI_BASE = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE') or 'http://127.0.0.1:8100/v1'
os.environ['OPENAI_BASE_URL'] = OPENAI_BASE
os.environ['OPENAI_API_BASE'] = OPENAI_BASE
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

MODEL = os.getenv('SKETCHCAD_MODEL') or 'openai/qwen3.8-flash-next'
if not MODEL.startswith('openai/'):
    MODEL = f'openai/{MODEL}'

CAD_BACKEND = os.getenv('CAD_BACKEND', 'cadcraft').lower()
CADCRAFT_CLI_CMD = (
    os.getenv('CADCRAFT_CLI_PATH')
    or ('/home/udhayan/Agent/cadcraft-main/target/debug/cadcraft-cli' if os.path.isfile('/home/udhayan/Agent/cadcraft-main/target/debug/cadcraft-cli') else None)
    or shutil.which('cadcraft-cli')
    or '/usr/bin/cadcraft-cli'
)
FREECAD_MCP_CMD = shutil.which('freecad-mcp') or str(Path(sys.executable).parent / 'freecad-mcp')

OUTPUT_DIR = Path("/home/udhayan/Agent/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_cadcraft_gui_proc = None

def open_in_cadcraft_gui(file_path: str = "") -> str:
    """Opens CADCraft desktop GUI application with the generated CAD file to showcase the output directly on screen.

    Args:
        file_path: Path to the DXF or DWG file in /home/udhayan/Agent/outputs/ (defaults to latest generated drawing).
    """
    global _cadcraft_gui_proc
    cadcraft_gui = shutil.which('cadcraft') or '/usr/bin/cadcraft'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not file_path or file_path in ["OUT.dxf", "OUT"]:
        dxfs = sorted(OUTPUT_DIR.glob("*.dxf"), key=os.path.getmtime, reverse=True)
        if dxfs:
            abs_path = str(dxfs[0])
        else:
            abs_path = str(OUTPUT_DIR / "drawing.dxf")
    elif not os.path.isabs(file_path):
        abs_path = str(OUTPUT_DIR / file_path)
    else:
        abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        return f"Error: CAD file not found at {abs_path}. Please save the drawing first using save(path='{abs_path}')."

    # Close previous GUI preview window if still running to cleanly refresh
    if _cadcraft_gui_proc is not None and _cadcraft_gui_proc.poll() is None:
        try:
            _cadcraft_gui_proc.terminate()
            _cadcraft_gui_proc.wait(timeout=1.0)
        except Exception:
            pass

    env = dict(os.environ)
    if 'DISPLAY' not in env:
        env['DISPLAY'] = ':0'
    uid = os.getuid()
    if 'WAYLAND_DISPLAY' not in env and os.path.exists(f'/run/user/{uid}/wayland-1'):
        env['WAYLAND_DISPLAY'] = 'wayland-1'

    try:
        _cadcraft_gui_proc = subprocess.Popen(
            [cadcraft_gui, abs_path],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        return f"Successfully opened CADCraft GUI with {abs_path} (PID {_cadcraft_gui_proc.pid}). The CADCraft application is now showcasing the drawing directly on your screen."
    except Exception as e:
        return f"Failed to open CADCraft GUI: {e}"

class Entity(BaseModel):
    id: str
    type: Literal['line', 'circle', 'arc']
    p1: Optional[list[float]] = None
    p2: Optional[list[float]] = None
    center: Optional[list[float]] = None
    radius: Optional[float] = None
    start_deg: Optional[float] = None
    end_deg: Optional[float] = None

class Relation(BaseModel):
    kind: str
    refs: list[str]
    value_mm: Optional[float] = None

class TextAnnotation(BaseModel):
    text: str
    position: list[float]  # [x, y] coordinates in mm
    height: Optional[float] = 3.0  # text height in mm
    rotation_deg: Optional[float] = 0.0

class PartComponent(BaseModel):
    id: str  # e.g. "part_1"
    name: str  # e.g. "Nominal Sharp Outer Envelope", "Central Bore & Cuts", "Corner Fillets"
    type: Literal['base_envelope', 'circular_feature', 'cutout_joint', 'fillet_round', 'annotation']
    description: str
    sharp_vertices: list[list[float]] = []  # nominal sharp corner coordinates referenced from (0,0)
    target_entity_ids: list[str] = []

class DrawingSpec(BaseModel):
    units: Literal['mm']
    entities: list[Entity]
    relations: list[Relation]
    annotations: list[TextAnnotation] = []
    parts: list[PartComponent] = []
    assumptions: list[str]
    unreadable: list[str]

class Discrepancy(BaseModel):
    issue_type: str
    description: str
    how_to_fix: str

class CriticReport(BaseModel):
    accuracy_score: float
    is_accurate: bool
    summary: str
    errors: list[Discrepancy]
    rectification_commands: list[str]

class BoundingBox(BaseModel):
    xmin: float
    xmax: float
    ymin: float
    ymax: float

class OuterLoopLine(BaseModel):
    id: str
    type: Literal['line']
    p1: list[float]
    p2: list[float]

class OuterLoopArc(BaseModel):
    id: str
    type: Literal['arc']
    center: list[float]
    radius: float
    start_angle: float
    end_angle: float
    sweep: float
    midpoint: list[float]
    p_start: list[float]
    p_end: list[float]

class InnerCircle(BaseModel):
    id: str
    type: Literal['circle']
    center: list[float]
    radius: float

class FilletSpec(BaseModel):
    corner: list[float]
    between: list[str]
    radius: float
    basis: Literal['STATED', 'ESTIMATED']
    expected_sweep: float

class SelfChecks(BaseModel):
    loop_closed: bool
    bbox_ok: bool
    arcs_ok: bool

class AnalystJsonSpec(BaseModel):
    units: Literal['mm'] = 'mm'
    h: Optional[float] = None
    origin: str = 'baseline, bottom-left of bounding box'
    bbox: BoundingBox
    stated_labels: list[str] = []
    outer_loop: list[Union[OuterLoopLine, OuterLoopArc]] = []
    inner_features: list[InnerCircle] = []
    fillets: list[FilletSpec] = []
    checks: SelfChecks
    uncertain: list[str] = []

ANALYST_PROMPT = """You are an expert mechanical CAD perception engineer.
Analyze the provided 2D engineering drawing image carefully.
Your task is to transcribe the visual geometry into a mathematically verified CAD specification for the Draftsman agent.

GROUNDING & DATUM:
1. Origin (0,0): bottom-left of the part bounding box. Y=0 at lowest horizontal edge, X=0 at leftmost outer contour edge. Units: mm.
2. Read every printed dimension (Ø, R, linear) exactly. Trace BOTH extension lines to identify the exact two points each dimension measures between.
3. Dimension to circle CENTER (e.g. "140" to hub center): the circle OUTER EDGE extends further by +radius beyond that center point.
4. Right-wall tangency: if a hub's rightmost edge is tangent to the right vertical wall, then cx_hub = x_right_wall - R_hub.
5. CAD angle convention: CCW from +X axis (0°=right, 90°=up, 180°=left, 270°=down).

PART NAME: Use snake_case (e.g. "linkage_bracket"). Never use "OUT", "cad_part", or screenshot filenames.

MANDATORY TOOL WORKFLOW — use python_tool for ALL math. Do not guess tangent points or arc centers:
1. Verify all dimension chains with python_tool.
2. Concave fillet between a LINE (at y=Y) and a CIRCLE (center C, radius R_circle) with blend radius R_blend:
   - Fillet center is on the AIR side: cy_fillet = Y + R_blend (above the line), cx_fillet = cx_circle - sqrt((R_blend + R_circle)^2 - (cy_fillet - cy_circle)^2)
   - Tangent on line: (cx_fillet, Y)
   - Tangent on circle: circle_center + R_circle * unit_vector_toward_fillet_center
   - Call arc_angles to get start/end degrees; verify via_point is ON the concave curve between the two tangent points.
3. Rounded lugs / tangent from baseline to circle: call tangent_from_point_to_circle to find exact tangent points.
4. For EVERY arc in outer_loop:
   - Specify p_start, via_point (midpoint ON the arc, on the SHORT side between p_start and p_end), p_end.
   - Verify: CCW sweep from start_deg to end_deg MUST be < 350°. If > 350°, via_point is on the wrong side — recompute it at the CCW midpoint angle.

OUTER LOOP — trace the closed outer boundary CCW (counterclockwise):
- Lines: {"id":"L1","type":"line","p1":[x1,y1],"p2":[x2,y2]}
- Arcs:  {"id":"A1","type":"arc","center":[cx,cy],"radius":r,"start_deg":d1,"end_deg":d2,"p_start":[sx,sy],"via_point":[mx,my],"p_end":[ex,ey]}

INNER FEATURES — all circular bores/bosses AND internal lines:
- Circle: {"id":"C1","type":"circle","center":[cx,cy],"radius":r,"style":"solid"}
- Line: {"id":"L2","type":"line","p1":[x1,y1],"p2":[x2,y2],"style":"solid"} or "dashed"

Call python_tool first, then output ONLY valid JSON:

```json
{
  "part_name": "<snake_case_name>",
  "units": "mm",
  "description": "<brief description>",
  "bbox": {"xmin": 0.0, "ymin": 0.0, "xmax": <width>, "ymax": <height>},
  "outer_loop": [
    {"id": "L1", "type": "line", "p1": [0.0, 0.0], "p2": [<x>, 0.0]}
  ],
  "inner_features": [
    {"id": "C1", "type": "circle", "center": [<cx>, <cy>], "radius": <r>, "style": "solid"}
  ],
  "dimensions": ["..."],
  "checks": {"loop_closed": true, "bbox_ok": true}
}
```
"""

DRAFTSMAN_PROMPT = """You are the expert CAD Draftsman. You receive the Technical CAD Drafting Directive from the Perception Agent.
Your job is to read this directive and directly perform LIVE 2D CAD DRAWING in FreeCAD software on the user's screen!

TECHNICAL CAD DRAFTING DIRECTIVE FROM PERCEPTION AGENT:
{cad_description}

(Fallback Reference: {drawing_spec})

MANDATORY STEP-BY-STEP LIVE DRAFTING IN FREECAD:

PHASE 1: INITIALIZE FREECAD DOCUMENT & 2D SKETCH
1. Call create_document(name="Drawing") to create a fresh document in FreeCAD.
2. Call create_sketch(doc_name="Drawing", name="Sketch", plane="XY_Plane") to create a 2D sketch on the XY plane.
   (Live drawing takes place inside this sketch named "Sketch" in document "Drawing").

PHASE 2: LIVE DRAWING OF SEPARATED PARTS IN ORDER
Follow the functional part decomposition in the Technical CAD Drafting Directive:
Order: Base Nominal Envelope -> Corner Fillets -> Circular Features & Bores -> Cuts & Slots.

Tool Call Reference for FreeCAD:
- Lines: add_sketch_line(doc_name="Drawing", sketch_name="Sketch", x1=x1, y1=y1, x2=x2, y2=y2)
- Arcs: add_sketch_arc(doc_name="Drawing", sketch_name="Sketch", center_x=cx, center_y=cy, radius=r, start_angle=deg1, end_angle=deg2)
- Circles: add_sketch_circle(doc_name="Drawing", sketch_name="Sketch", center_x=cx, center_y=cy, radius=r)
- Scripting / Batching / Constraints: execute_python(code="...")

PHASE 3: HOW TO APPLY FILLETS (SHARP EDGES FIRST -> THEN FILLET TOOL)
MANDATORY CAD RULE: Never attempt to guess or draw freehand curves for filleted corners!
Follow this standard 2-step CAD engineering procedure:
Step 3A (Sharp Nominal Edges):
   - Identify the sharp intersection corner vertex C = [cx, cy] where two nominal straight edges meet.
Step 3B (Fillet Tool):
   - Call fillet_two_lines(corner=[cx, cy], away1=[dx1, dy1], away2=[dx2, dy2], radius=R).
   - The tool returns:
     * tangent_point_1: [t1_x, t1_y] (where Line 1 ends)
     * tangent_point_2: [t2_x, t2_y] (where Line 2 begins)
     * center: [arc_cx, arc_cy]
     * start_deg, end_deg: exact CCW arc angles.
   - Draw Line 1 up to tangent_point_1:
     add_sketch_line(doc_name="Drawing", sketch_name="Sketch", x1=p_start[0], y1=p_start[1], x2=t1_x, y2=t1_y)
   - Insert the fillet arc:
     add_sketch_arc(doc_name="Drawing", sketch_name="Sketch", center_x=arc_cx, center_y=arc_cy, radius=R, start_angle=start_deg, end_angle=end_deg)
   - Draw Line 2 continuing from tangent_point_2:
     add_sketch_line(doc_name="Drawing", sketch_name="Sketch", x1=t2_x, y1=t2_y, x2=p_next[0], y2=p_next[1])
This replaces the sharp corner with a perfectly watertight, tangent curve blend with zero gap!

PHASE 4: CIRCLE CUTS AND JOINTS (STEP-BY-STEP)
When a circular bore or feature has a cutout, notch, keyway, or meeting joint:
1. Call circle_line_intersections(center=[cx, cy], radius=R, p1=..., p2=...) to find exact penetration points on the circle circumference.
2. Call arc_angles(center=[cx, cy], p_start=..., p_end=..., via_point=...) to get CCW start_deg and end_deg for the remaining portion of the circle.
3. Draw the cut lines and kept arc into "Sketch", ensuring endpoints connect with zero gap.

PHASE 5: RECOMPUTE, FIT ALL & CAPTURE LIVE SCREENSHOT
1. Call recompute_document(doc_name="Drawing") to update solver state.
2. Call fit_all(doc_name="Drawing") so the entire live drawing is framed perfectly in FreeCAD's 3D GUI view.
3. Call get_screenshot(view_angle="Top", doc_name="Drawing") to capture the 2D rendering for the Visual Critic.
4. Conclude: Summarize what was drawn live in FreeCAD.
Rules: never invent tool names. Coordinates come strictly from the technical directive. At most 40 tool calls.

DECICAD HARNESS PRINCIPLES:
- Spatial Contract: Right-handed Cartesian coordinates (+X right, +Y back/up, +Z up), lengths in mm, angles in deg.
- State Grounding: All actions execute against the active FreeCAD document. Never clear or recreate from scratch when recovering; build incrementally upon existing stable sketch geometry.
- Two-Tier Observation: Read the immediate execution feedback of each tool call. When solver or geometry state is uncertain, use active hierarchical inspection (list_objects, inspect_object) rather than guessing.
"""

CRITIC_PROMPT = """You are the Visual Critic / Inspector of the CAD pipeline.
Compare the current FreeCAD 2D drawing screenshot directly against the ORIGINAL TECHNICAL DRAWING IMAGE uploaded by the user and Technical CAD Directive:
{cad_description}
(Fallback Reference: {drawing_spec})

Your Goal: Strictly inspect the visual geometry in FreeCAD against the user's drawing image and specification. Target accuracy is 98% and above.

Procedure:
1. Inspect the FreeCAD drawing: call fit_all(doc_name="Drawing"), then call get_screenshot(view_angle='Top', doc_name="Drawing") to inspect the visual rendering.
2. Conduct a strict visual audit across 5 mandatory checks:
   - Check A (Base & Fillets): Base line and bottom corner fillets.
   - Check B (Side Profiles): Side contours, vertical edges, and tangent transitions.
   - Check C (Main Outer Head): Top circular arcs and heads.
   - Check D (Protrusions & Angles): Angled tabs, ears, and sloped transitions.
   - Check E (Inner Features): Central bore holes and keyway slot/cutout joints.
3. Strict Scoring Rules:
   - If contours are open, features are missing, or geometry contradicts the dimensions: ACCURACY SCORE CANNOT EXCEED 85.0%.
   - Score >= 98.0% ONLY IF ALL features are completely drawn, topologically connected, and match the dimensions.
4. If accuracy_score >= 98.0:
   - Call the exit_loop tool to complete the loop!
   - Set is_accurate to True and summarize that the drawing meets the quality target (>= 98%).
5. If accuracy_score < 98.0:
   - Do NOT call exit_loop.
   - Set is_accurate to False.
   - In errors: detail each missing check with structured issue_type ('missing_feature', 'open_contour', 'geometry_conflict', 'dimension_mismatch', or 'invalid_state').
   - In rectification_commands: provide exact FreeCAD commands (add_sketch_line, add_sketch_arc, or execute_python) for the Refiner to execute incrementally in the active sketch.
"""

REFINER_PROMPT = """You are the CAD Refiner. Your job is to rectify errors reported by the Critic and update the FreeCAD drawing live.
Critic Report:
{critic_report}

Technical CAD Directive:
{cad_description}
(Fallback Reference: {drawing_spec})

Procedure:
1. Review the errors and rectification_commands from the Critic Report.
2. Apply the corrections in FreeCAD:
   - Execute the Critic's rectification commands to add the missing geometry (arcs, lines, circles, slots) into the active sketch.
   - Follow DeciCAD state grounding: modify the persistent active sketch incrementally; never recreate the sketch from scratch.
   - Ensure new geometry endpoints connect properly to form closed loops.
3. Call recompute_document(doc_name="Drawing"), fit_all(doc_name="Drawing"), and call get_screenshot(view_angle='Top', doc_name="Drawing") to capture the updated drawing.
4. Summarize what was added/fixed so the Critic can re-evaluate on the next iteration of the loop.
Rules: never invent tool names. Follow the Critic's rectification commands strictly."""

DRAFTSMAN_CADCRAFT_PROMPT = """You draw 2D engineering profiles directly in CADCraft using MCP tools and mathematical geometry helpers.
You receive a structured specification from the Analyst, and you also see the user's ORIGINAL drawing image in the conversation.
All coordinates are in millimeters anchored from origin datum (0.0, 0.0).

SPECIFICATION FROM ANALYST:
{drawing_spec}

OUTPUT DIRECTORY & DYNAMIC NAMING:
- Dedicated Folder: All CAD files and rendered images MUST be saved in `/home/udhayan/Agent/outputs/`.
- File Naming: Use `{part_name}` (e.g. "mounting_bracket", "linkage_bracket"):
  * DXF path: `/home/udhayan/Agent/outputs/{part_name}.dxf`
  * Rendered preview: `/home/udhayan/Agent/outputs/{part_name}_rendered.png`
- NEVER save as generic `OUT.dxf` and NEVER use raw screenshot filenames!

CADCRAFT COMMAND REFERENCE:
- Create drawing: new_drawing(metric=True)
- Load linetypes (for hidden/center lines): execute(command="linetype", params={"load": "*"})
- Create layer: execute(command="layer.new", params={"name": "VISIBLE", "color": "white", "current": True})
  * For dashed hidden lines: execute(command="layer.new", params={"name": "HIDDEN", "color": "cyan", "linetype": "DASHED", "current": True})
  * For centerlines: execute(command="layer.new", params={"name": "CENTER", "color": "red", "linetype": "CENTER", "current": True})
- Draw straight line(s): execute(command="line", params={"points": [[x1, y1], [x2, y2], ...]})
- Draw closed polyline: execute(command="pline", params={"vertices": [[x1, y1], ...], "closed": True})
- Draw unambiguous 3-point arc (PREFERRED whenever p_start, via_point, p_end are known):
  execute(command="arc", params={"p1": [sx, sy], "p2": [mx, my], "p3": [ex, ey]})
- Draw center-radius CCW arc (ONLY when start/end angles were verified via `arc_angles`):
  execute(command="arc", params={"center": [cx, cy], "radius": r, "start": deg1, "end": deg2})
- Draw circular bore/boss: execute(command="circle", params={"center": [cx, cy], "radius": r})
- Apply fillet (works on lines, arcs, and circles): add_fillet(entity1="handle1", entity2="handle2", radius=r)
- Add dimensions (on DIMENSIONS layer):
  * execute(command="dimlinear", params={"p1": [x1, y1], "p2": [x2, y2], "at": [ax, ay]})
  * execute(command="dimradius", params={"center": [cx, cy], "point": [px, py]})
  * execute(command="dimdiameter", params={"center": [cx, cy], "point": [px, py]})
- Save drawing: save(path="/home/udhayan/Agent/outputs/{part_name}.dxf")
- Render image preview: render(fit=True, path="/home/udhayan/Agent/outputs/{part_name}_rendered.png")
- Open live GUI showcase: open_in_cadcraft_gui(file_path="/home/udhayan/Agent/outputs/{part_name}.dxf")

GEOMETRIC SANITY & PROCEDURE:
1. Initialize: Call new_drawing(metric=True), load linetypes `linetype(load="*")`, and create layer `VISIBLE` (`white`, `current=True`).
2. Verify Before Drawing: Check every arc and tangent transition against the original drawing image!
   - If an arc in the spec is a concave blend (like R89) whose center lies inside the solid body instead of outside on the air side, RECALCULATE it right now using `tangent_arc_line_circle` + `arc_angles`!
   - Prefer 3-point `arc(p1=p_start, p2=via_point, p3=p_end)` so concave curves NEVER flip upside-down into domes.
3. Draw Outer Boundary on `VISIBLE`: Ensure sequential lines and arcs form a closed watertight perimeter with zero gaps.
4. Draw Concentric Bores, Bosses & Internal Solid Ribs on `VISIBLE`: Draw ALL concentric circles (e.g., both outer hub ring AND inner bore) and solid rib lines.
5. Draw Hidden / Dashed Features on `HIDDEN` (`linetype="DASHED"`): Draw internal hidden walls and dashed slots.
6. Apply Small Fillets & Add Centerlines (`CENTER` layer) + Key Dimensions (`DIMENSIONS` layer).
7. Save, Render & Showcase:
   Call save(path="/home/udhayan/Agent/outputs/{part_name}.dxf"),
   render(fit=True, path="/home/udhayan/Agent/outputs/{part_name}_rendered.png"),
   and open_in_cadcraft_gui(file_path="/home/udhayan/Agent/outputs/{part_name}.dxf").
8. Conclude with a concise summary listing `/home/udhayan/Agent/outputs/{part_name}.dxf`.
"""

CRITIC_CADCRAFT_PROMPT = """You are the strict Visual Critic & Geometric Auditor of the CAD pipeline.
You MUST visually inspect the rendered CADCraft drawing side-by-side against the ORIGINAL TECHNICAL DRAWING IMAGE uploaded by the user and specification:
{cad_description}
(Fallback Reference: {drawing_spec})

CRITICAL STEP 1 (MANDATORY BEFORE SCORING):
You DO NOT yet see the current CADCraft drawing until you call `render(...)`!
On your FIRST turn of each Critic iteration, ONLY call tools:
- `render(fit=True, path="/home/udhayan/Agent/outputs/{part_name}_rendered.png")`
- `inspect_drawing(entities=False)`
Do NOT output your final JSON score or call `exit_loop` until AFTER `render` returns the rendered PNG image to you!

STEP 2 (VISUAL & GEOMETRIC AUDIT AFTER SEEING THE RENDERED PNG):
Compare `[RENDERED CADCRAFT OUTPUT IMAGE]` directly against the user's ORIGINAL drawing image:
- Check 1 (Silhouette & Arc Curvature Direction): Do all outer curves bend in the exact same direction as the blueprint? (e.g., concave valley fillets scoop INWARD; NO upside-down convex domes arching across empty space; NO invented arcs not in the drawing; left/right lugs and sloped tangent lines connect smoothly).
- Check 2 (Concentric Circles & Bores): Are ALL concentric circles drawn (e.g., outer hub AND inner bore, outer bosses AND inner holes)? Are proportions and vertical/horizontal alignments faithful?
- Check 3 (Closure & Tangency): Are outer boundaries closed with zero dangling segments, overlaps, or sharp kinks at tangent joints?
- Check 4 (Internal Ribs, Hidden Dashed Lines & Centerlines): Are internal blocks, vertical ribs, dashed hidden outlines, and centerlines present cleanly?
- Check 5 (Dimensions & Dynamic Naming): Are key dimensions present, and is the output saved to `/home/udhayan/Agent/outputs/{part_name}.dxf` (NOT `OUT.dxf`)?

SCORING RULES (BE STRICT AND HONEST):
- If ANY arc curves the wrong way (e.g., dome instead of concave valley), a boundary is broken/open, a concentric bore/hole is missing, or extra spurious geometry exists: accuracy_score MUST BE < 75.0 (set `is_accurate: false`).
- Score >= 95.0 (`is_accurate: true`) ONLY IF the rendered image visually matches the blueprint silhouette, arc curvatures, holes, proportions, and watertightness.

ACTIONS:
- If `accuracy_score >= 95.0` (AFTER visually inspecting the rendered PNG):
  1. Call `open_in_cadcraft_gui(file_path="/home/udhayan/Agent/outputs/{part_name}.dxf")`.
  2. Call `exit_loop()`.
  3. Output final JSON: `{"accuracy_score": <score>, "is_accurate": true, "summary": "...", "errors": [], "rectification_commands": []}`.
- If `accuracy_score < 95.0`:
  1. Do NOT call `exit_loop`.
  2. Set `is_accurate: false`, list each specific geometric/visual defect in `errors`, and provide clear `rectification_commands` (including `delete_entity` handles from `query_entities` or `"rebuild_clean"` when major silhouette arcs are wrong) so the Refiner can fix every single defect! Output ONLY valid JSON.
"""

REFINER_CADCRAFT_PROMPT = """You are the CAD Refiner. Your job is to rectify every geometric and visual defect reported by the Critic.
Critic Report:
{critic_report}

Technical CAD Directive:
{cad_description}
(Fallback Reference: {drawing_spec})

PROCEDURE:
1. Review `errors` and `rectification_commands` from the Critic Report, and compare against the original drawing image.
2. Inspect entity handles with `query_entities()` if needed.
3. Apply accurate geometric repairs using `GEOMETRY_TOOLS` (`tangent_arc_line_circle`, `tangent_from_point_to_circle`, `arc_angles`, `fillet_two_lines`) — NEVER guess arc angles or tangent endpoints:
   - For localized fixes (missing circles/bores, stray lines, single broken arc): delete broken entities with `delete_entity(id=...)` / `execute(command="erase", params={"handles": [...]})` and draw the mathematically verified replacements (use 3-point `arc(p1, p2, p3)` for unambiguous curvature!).
   - If the Critic flags major silhouette inversion (multiple wrong outer arcs or disconnected base contour), cleanly reset with `new_drawing(metric=True)` and rebuild the verified profile using exact `GEOMETRY_TOOLS` coordinates, layers (`VISIBLE`, `HIDDEN`, `CENTER`, `DIMENSIONS`), and 3-point arcs!
4. Save & Render:
   - `save(path="/home/udhayan/Agent/outputs/{part_name}.dxf")`
   - `render(fit=True, path="/home/udhayan/Agent/outputs/{part_name}_rendered.png")`
   - `open_in_cadcraft_gui(file_path="/home/udhayan/Agent/outputs/{part_name}.dxf")`
5. Summarize the exact fixes made so the Critic can visually inspect the updated render.
"""

import asyncio
import io
import json
import time
import litellm
from PIL import Image, ImageEnhance
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.workflow._retry_config import RetryConfig

# Performance settings for Qwen vLLM endpoint (accounts for Tailscale RTT)
litellm.request_timeout = 180.0
litellm.disable_cost_tracking = True
os.environ['LITELLM_REQUEST_TIMEOUT'] = '180'
os.environ['LITELLM_NUM_RETRIES'] = '0'
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'

extra_body = {}
if 'qwen' in MODEL.lower():
    extra_body['chat_template_kwargs'] = {'enable_thinking': False}

class QwenLiteLlm(LiteLlm):
    """LiteLlm wrapper for Qwen vLLM endpoints preserving native auto tool calling."""
    pass

active_model = QwenLiteLlm(
    model=MODEL,
    api_base=OPENAI_BASE,
    api_key=OPENAI_KEY,
    timeout=180,
    extra_body=extra_body if extra_body else None
)

_last_call_time = 0.0
PACE_SECONDS = float(os.getenv('SKETCHCAD_PACE_SECONDS', '0.0'))
_entity_id_map: dict[str, str] = {}
_last_created_handle: Optional[str] = None
_handles_history: list[str] = []

def sanitize_json_output(text: str, default_schema: Optional[type[BaseModel]] = None) -> str:
    """Extracts valid JSON payload and discards conversational greetings or preambles."""
    if not text or not text.strip():
        if default_schema == DrawingSpec:
            return json.dumps({
                "units": "mm",
                "entities": [{"id": "e1", "type": "line", "p1": [0.0, 0.0], "p2": [65.0, 0.0]}],
                "relations": [],
                "annotations": [],
                "assumptions": ["Origin anchored at (0,0)"],
                "unreadable": []
            })
        elif default_schema == CriticReport:
            return json.dumps({
                "accuracy_score": 0.0,
                "is_accurate": False,
                "summary": "Critic returned empty response; visual inspection required.",
                "errors": [{"issue_type": "invalid_state", "description": "Empty Critic response", "how_to_fix": "Call render and inspect geometry"}],
                "rectification_commands": []
            })
        return "{}"

    text = text.strip()
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end != -1:
            text = text[start:end].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    if default_schema == DrawingSpec:
        return json.dumps({
            "units": "mm",
            "entities": [{"id": "e1", "type": "line", "p1": [0.0, 0.0], "p2": [65.0, 0.0]}],
            "relations": [],
            "annotations": [],
            "parts": [],
            "assumptions": ["Origin anchored at (0,0)"],
            "unreadable": []
        })
    elif default_schema == CriticReport:
        return json.dumps({
            "accuracy_score": 0.0,
            "is_accurate": False,
            "summary": f"Critic failed to output valid JSON evaluation: {text[:120]}",
            "errors": [{"issue_type": "invalid_state", "description": "Non-JSON Critic response", "how_to_fix": "Inspect rendered image and output valid CriticReport JSON"}],
            "rectification_commands": []
        })
    return text

def _normalize_execute_args(args: dict) -> dict:
    """Normalizes parameter structure for CADCraft MCP commands."""
    if not isinstance(args, dict):
        return args
    params = dict(args.get('params') or {})
    cmd = args.get('command') or params.get('command')
    if cmd == 'line':
        p1 = args.get('p1') if args.get('p1') is not None else params.get('p1')
        p2 = args.get('p2') if args.get('p2') is not None else params.get('p2')
        pts = args.get('points') if args.get('points') is not None else params.get('points')
        if p1 is not None and p2 is not None:
            params['points'] = [p1, p2]
        elif pts is not None:
            params['points'] = pts
    elif cmd == 'arc':
        p1 = args.get('p1') if args.get('p1') is not None else params.get('p1')
        p2 = args.get('p2') if args.get('p2') is not None else params.get('p2')
        p3 = args.get('p3') if args.get('p3') is not None else params.get('p3')
        if p1 is not None and p2 is not None and p3 is not None:
            params['p1'] = p1
            params['p2'] = p2
            params['p3'] = p3
        c = args.get('center') if args.get('center') is not None else params.get('center')
        r = args.get('radius') if args.get('radius') is not None else params.get('radius')
        s = None
        for k in ['start_angle', 'start']:
            if args.get(k) is not None:
                s = args.get(k)
                break
            if params.get(k) is not None:
                s = params.get(k)
                break
        e = None
        for k in ['end_angle', 'end']:
            if args.get(k) is not None:
                e = args.get(k)
                break
            if params.get(k) is not None:
                e = params.get(k)
                break
        if c is not None:
            params['center'] = c
        if r is not None:
            params['radius'] = r
        if s is not None:
            params['start'] = s
        if e is not None:
            params['end'] = e
    elif cmd == 'circle':
        c = args.get('center') if args.get('center') is not None else params.get('center')
        r = args.get('radius') if args.get('radius') is not None else params.get('radius')
        if c is not None:
            params['center'] = c
        if r is not None:
            params['radius'] = r
    elif cmd in ['erase', 'delete']:
        h = args.get('handles') if args.get('handles') is not None else args.get('handle')
        if h is None:
            h = params.get('handles') if params.get('handles') is not None else params.get('handle')
        if h is not None:
            params['handles'] = [h] if isinstance(h, (str, int)) else h

    return {'command': cmd, 'params': params}


def _normalize_tool_call_args(fn_name: str, args: dict) -> dict:
    """Ensures tool parameters match Python tool function signatures."""
    if not isinstance(args, dict):
        return args
    if fn_name == 'execute':
        return _normalize_execute_args(args)
    elif fn_name == 'add_fillet':
        e1 = args.get('entity1') or args.get('h1') or (args.get('between', [None, None])[0] if isinstance(args.get('between'), list) else None)
        e2 = args.get('entity2') or args.get('h2') or (args.get('between', [None, None])[1] if isinstance(args.get('between'), list) else None)
        r = args.get('radius', 0.5)
        out = {'entity1': str(e1) if e1 is not None else '', 'entity2': str(e2) if e2 is not None else '', 'radius': float(r)}
        if args.get('pick1') or args.get('p1'):
            out['pick1'] = args.get('pick1') or args.get('p1')
        if args.get('pick2') or args.get('p2'):
            out['pick2'] = args.get('pick2') or args.get('p2')
        return out
    elif fn_name in ['read_entity', 'delete_entity']:
        ident = args.get('id') or args.get('handle') or ''
        return {'id': str(ident)}
    return args


def _extract_valid_json_spec(text: str) -> tuple[Optional[dict], Optional[str]]:
    """Extracts the single valid Analyst JSON spec object from text, even if preceded by working or multiple runs."""
    if not text:
        return None, None
    # 1. Search inside ```json ... ``` code fence
    if "```json" in text:
        parts = text.split("```json")
        for chunk in reversed(parts[1:]):
            sub = chunk.split("```")[0].strip()
            try:
                p = json.loads(sub)
                if isinstance(p, dict) and ("outer_loop" in p or "bbox" in p or "entities" in p):
                    return p, sub
            except Exception:
                pass

    # 2. Search for balanced braces from the end of the text backwards
    for end_idx in [i for i, c in enumerate(text) if c == '}'][::-1]:
        depth = 0
        for start_idx in range(end_idx, -1, -1):
            if text[start_idx] == '}':
                depth += 1
            elif text[start_idx] == '{':
                depth -= 1
                if depth == 0:
                    cand = text[start_idx:end_idx + 1]
                    try:
                        p = json.loads(cand)
                        if isinstance(p, dict) and ("outer_loop" in p or "bbox" in p or "entities" in p):
                            return p, cand
                    except Exception:
                        pass
                    break
    return None, None


async def _sanitize_analyst_output(callback_context, llm_response):
    """Sanitizes analyst output into an actionable Technical CAD Drafting Directive and extracts JSON spec."""
    await _parse_and_inject_tool_calls(callback_context, llm_response)

    parts = getattr(getattr(llm_response, 'content', None), 'parts', None) or []
    has_tool_calls = any(getattr(p, 'function_call', None) for p in parts)

    state_calls = 0
    if callback_context and hasattr(callback_context, 'state'):
        state_calls = int(callback_context.state.get('analyst_tool_calls', 0))
        if has_tool_calls:
            state_calls += 1
            callback_context.state['analyst_tool_calls'] = state_calls

    output_text = ""
    for part in parts:
        t = getattr(part, 'text', None)
        if t:
            if t.startswith("I am Qwen."):
                t = t[len("I am Qwen."):].strip()
            part.text = t
            output_text += t + "\n"

    cleaned_text = output_text.strip()
    parsed_spec, spec_json_str = _extract_valid_json_spec(cleaned_text)

    if not spec_json_str:
        spec_json_str = cleaned_text

    # Store in session state whenever valid JSON is available (or as fallback text)
    if callback_context and hasattr(callback_context, 'state'):
        try:
            if parsed_spec or not has_tool_calls:
                callback_context.state['drawing_spec'] = spec_json_str
                callback_context.state['cad_description'] = cleaned_text
            if parsed_spec and isinstance(parsed_spec, dict):
                p_name = parsed_spec.get('part_name') or 'mechanical_part'
                p_slug = re.sub(r'[^a-zA-Z0-9]+', '_', str(p_name).lower()).strip('_') or 'mechanical_part'
                callback_context.state['part_name'] = p_slug
            else:
                callback_context.state.setdefault('part_name', 'mechanical_part')
        except Exception:
            pass

    # Allow Analyst up to 12 Python/geometry tool-calling turns before forcing final spec completion
    if llm_response and getattr(llm_response, 'content', None) and parts:
        if has_tool_calls and state_calls < 12 and parsed_spec is None:
            return None
        if parsed_spec is not None or state_calls >= 12:
            non_tool_parts = [
                part for part in parts
                if not getattr(part, 'function_call', None)
            ]
            if non_tool_parts:
                llm_response.content.parts = non_tool_parts
            else:
                llm_response.content.parts = [types.Part.from_text(text=spec_json_str or "{}")]
            if callback_context and hasattr(callback_context, 'state'):
                callback_context.state['analyst_tool_calls'] = 0

    return None

_sanitize_drawing_spec_output = _sanitize_analyst_output

async def _sanitize_critic_output(callback_context, llm_response):
    """Sanitizes critic output, ensuring visual render occurs before evaluation or exit_loop."""
    await _parse_and_inject_tool_calls(callback_context, llm_response)
    if not (llm_response and getattr(llm_response, 'content', None) and getattr(llm_response.content, 'parts', None)):
        return None

    parts = list(llm_response.content.parts)
    has_rendered = False
    part_name = 'mechanical_part'
    if callback_context and hasattr(callback_context, 'state'):
        has_rendered = bool(callback_context.state.get('critic_rendered_this_iter', False))
        part_name = callback_context.state.get('part_name', 'mechanical_part')

    # Check tool calls
    fn_calls = [getattr(p, 'function_call', None) for p in parts if getattr(p, 'function_call', None)]
    fn_names = [fc.name for fc in fn_calls]

    # Guard: prevent Critic from calling exit_loop before render() has visually shown the CAD output
    if 'exit_loop' in fn_names and not has_rendered:
        filtered_parts = [p for p in parts if not (getattr(p, 'function_call', None) and p.function_call.name == 'exit_loop')]
        if 'render' not in fn_names and 'get_screenshot' not in fn_names:
            render_tool = 'render' if CAD_BACKEND == 'cadcraft' else 'get_screenshot'
            render_args = (
                {'fit': True, 'path': str(OUTPUT_DIR / f"{part_name}_rendered.png")}
                if CAD_BACKEND == 'cadcraft'
                else {'view_angle': 'Top', 'doc_name': 'Drawing'}
            )
            filtered_parts.append(types.Part.from_function_call(name=render_tool, args=render_args))
        parts = filtered_parts
        fn_calls = [getattr(p, 'function_call', None) for p in parts if getattr(p, 'function_call', None)]
        fn_names = [fc.name for fc in fn_calls]

    has_pending_inspection_tools = any(name != 'exit_loop' for name in fn_names)

    new_parts = []
    for part in parts:
        t = getattr(part, 'text', None)
        if t and not has_pending_inspection_tools:
            sanitized_txt = sanitize_json_output(t, default_schema=CriticReport)
            # Also block false positive is_accurate=True if Critic hasn't rendered yet
            if not has_rendered:
                try:
                    rep = json.loads(sanitized_txt)
                    if isinstance(rep, dict) and rep.get('is_accurate'):
                        rep['is_accurate'] = False
                        rep['accuracy_score'] = min(float(rep.get('accuracy_score', 0.0)), 50.0)
                        rep['summary'] = "Visual inspection incomplete: render() has not been evaluated yet."
                        sanitized_txt = json.dumps(rep)
                except Exception:
                    pass
            part.text = sanitized_txt
        new_parts.append(part)

    llm_response.content.parts = new_parts
    return None

async def _parse_and_inject_tool_calls(callback_context, llm_response):
    """Parses raw Qwen XML <tool_call> tags and transforms them into native ADK FunctionCall parts."""
    if not (llm_response and getattr(llm_response, 'content', None) and getattr(llm_response.content, 'parts', None)):
        return None

    new_parts = []
    has_converted_tools = False

    for part in llm_response.content.parts:
        # Check existing function_call
        if getattr(part, 'function_call', None) and part.function_call:
            fn_name = part.function_call.name
            if fn_name.startswith('functions.'):
                fn_name = fn_name[len('functions.'):]
                part.function_call.name = fn_name
            norm_args = _normalize_tool_call_args(fn_name, dict(part.function_call.args or {}))
            part.function_call.args = norm_args
            new_parts.append(part)
            continue

        text = getattr(part, 'text', None)
        if text and '<tool_call>' in text:
            tool_blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
            remaining_text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
            if remaining_text:
                new_parts.append(types.Part.from_text(text=remaining_text))

            for block in tool_blocks:
                fn_match = re.search(r'<function=(.*?)>', block)
                if fn_match:
                    fn_name = fn_match.group(1).strip()
                    if fn_name.startswith('functions.'):
                        fn_name = fn_name[len('functions.'):]
                    if fn_name == 'show_drawing':
                        fn_name = 'inspect_drawing'

                    params = {}
                    param_matches = re.findall(r'<parameter=(.*?)>(.*?)</parameter>', block, re.DOTALL)
                    for p_name, p_val in param_matches:
                        p_name = p_name.strip()
                        p_val = p_val.strip()
                        if p_val.lower() == 'true':
                            params[p_name] = True
                        elif p_val.lower() == 'false':
                            params[p_name] = False
                        else:
                            try:
                                params[p_name] = json.loads(p_val)
                            except Exception:
                                try:
                                    params[p_name] = float(p_val) if '.' in p_val else int(p_val)
                                except Exception:
                                    params[p_name] = p_val
                    if fn_name == 'computer_use' and 'function' in params:
                        fn_name = params.pop('function')
                    params.pop('action', None)
                    params = _normalize_tool_call_args(fn_name, params)
                    new_parts.append(types.Part.from_function_call(name=fn_name, args=params))
                    has_converted_tools = True
                else:
                    try:
                        data = json.loads(block.strip())
                        if 'name' in data:
                            name = data['name']
                            if name.startswith('functions.'):
                                name = name[len('functions.'):]
                            if name == 'show_drawing':
                                name = 'inspect_drawing'
                            norm_args = _normalize_tool_call_args(name, data.get('arguments', {}))
                            new_parts.append(types.Part.from_function_call(name=name, args=norm_args))
                            has_converted_tools = True
                    except Exception:
                        pass
        else:
            new_parts.append(part)

    if has_converted_tools:
        llm_response.content.parts = new_parts
    return None


def _detect_and_summarize_image(data) -> str:
    """Extracts basic image info without injecting false geometry templates."""
    try:
        import base64
        if isinstance(data, str):
            data = base64.b64decode(data)
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        return f"Input Technical Drawing: 2D mechanical engineering drawing with resolution {w}x{h} px."
    except Exception:
        return "Input Technical Drawing: 2D mechanical profile with millimeter dimensions."

def _detect_and_summarize_pdf(data) -> str:
    """Renders PDF page to image and extracts text to avoid /v1/files 404 on local vLLM."""
    try:
        import base64
        import tempfile
        if isinstance(data, str):
            data = base64.b64decode(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "doc.pdf")
            with open(pdf_path, "wb") as f:
                f.write(data)

            extracted_text = ""
            try:
                res = subprocess.run(["pdftotext", "-f", "1", "-l", "2", pdf_path, "-"], capture_output=True, text=True, timeout=5)
                if res.stdout:
                    extracted_text = res.stdout.strip()[:400]
            except Exception:
                pass

            out_prefix = os.path.join(tmpdir, "page")
            subprocess.run(["pdftoppm", "-png", "-r", "150", "-f", "1", "-l", "1", pdf_path, out_prefix], check=True, timeout=5)
            pngs = [os.path.join(tmpdir, x) for x in os.listdir(tmpdir) if x.endswith(".png")]
            if pngs:
                with open(pngs[0], "rb") as pf:
                    img_summary = _detect_and_summarize_image(pf.read())
                    if extracted_text:
                        return f"{img_summary}\n- Document Notes: {extracted_text}"
                    return img_summary
    except Exception:
        pass
    return "Input Technical Drawing: 2D mechanical profile with millimeter dimensions."

def _prepare_image_bytes(raw_data: Any, enhance_blueprint: bool = False, max_dim: int = 2048) -> Optional[bytes]:
    """Normalizes image bytes, upscaling/enhancing small blueprints or bounding max resolution."""
    try:
        if isinstance(raw_data, str):
            import base64
            img_bytes = base64.b64decode(raw_data)
        else:
            img_bytes = bytes(raw_data)
        pil_img = Image.open(io.BytesIO(img_bytes))
        if pil_img.mode not in ('RGB', 'L'):
            pil_img = pil_img.convert('RGB')
        w, h = pil_img.size
        longest = max(w, h)
        changed = False
        if enhance_blueprint and longest < 1300:
            scale = min(2.0, 1550.0 / max(1, longest))
            pil_img = pil_img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
            pil_img = ImageEnhance.Contrast(pil_img).enhance(1.25)
            pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.4)
            changed = True
        elif longest > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            changed = True
        if not changed and isinstance(raw_data, (bytes, bytearray)):
            return img_bytes
        buf = io.BytesIO()
        pil_img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception:
        return None


async def _pace_and_preprocess(callback_context, llm_request):
    """Paces requests, enhances blueprint legibility, and delivers rendered CAD screenshots to the VLM."""
    global _last_call_time
    if PACE_SECONDS > 0:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < PACE_SECONDS:
            await asyncio.sleep(PACE_SECONDS - elapsed)
        _last_call_time = time.time()

    part_name = 'mechanical_part'
    agent_name = getattr(callback_context, 'agent_name', '') if callback_context else ''
    if callback_context and hasattr(callback_context, 'state'):
        callback_context.state.setdefault('part_name', 'mechanical_part')
        callback_context.state.setdefault('drawing_spec', '{}')
        callback_context.state.setdefault('cad_description', '')
        callback_context.state.setdefault('critic_report', '{}')
        part_name = callback_context.state.get('part_name') or 'mechanical_part'

    latest_render_bytes: Optional[bytes] = None
    saw_render_in_trailing_tools = False

    if llm_request and getattr(llm_request, 'contents', None):
        num_contents = len(llm_request.contents)
        for c_idx, content in enumerate(llm_request.contents):
            if getattr(content, 'parts', None):
                new_parts = []
                for part in content.parts:
                    if getattr(part, 'inline_data', None) and part.inline_data:
                        mime = part.inline_data.mime_type or ''
                        if mime == 'application/pdf' or mime.endswith('/pdf'):
                            summary = _detect_and_summarize_pdf(part.inline_data.data)
                            new_parts.append(types.Part.from_text(text=f"\n[DRAWING SPECIFICATION EXTRACTED FROM PDF]:\n{summary}\n"))
                            continue
                        elif mime.startswith('image/'):
                            processed = _prepare_image_bytes(part.inline_data.data, enhance_blueprint=True, max_dim=2048)
                            if processed:
                                part.inline_data.data = processed
                                part.inline_data.mime_type = 'image/png'
                            new_parts.append(part)
                            continue
                        else:
                            new_parts.append(types.Part.from_text(text="\n[DRAWING SPECIFICATION EXTRACTED FROM ATTACHMENT]:\nTechnical CAD Drawing with millimeter dimensions.\n"))
                            continue
                    elif getattr(part, 'function_response', None) and part.function_response:
                        fn_name = getattr(part.function_response, 'name', '')
                        resp = getattr(part.function_response, 'response', None)
                        if fn_name == 'new_drawing':
                            _entity_id_map.clear()
                            _handles_history.clear()
                            if callback_context and hasattr(callback_context, 'state'):
                                callback_context.state['critic_rendered_this_iter'] = False
                        elif fn_name == 'execute' and isinstance(resp, dict):
                            handles = resp.get('handles') or ([resp['handle']] if 'handle' in resp else [])
                            for h in handles:
                                _last_created_handle = str(h)
                                _handles_history.append(str(h))
                        if isinstance(resp, dict) and 'content' in resp and isinstance(resp['content'], list):
                            sanitized = []
                            for item in resp['content']:
                                if isinstance(item, dict) and item.get('type') == 'image' and 'data' in item:
                                    out_img = str(OUTPUT_DIR / f"{part_name}_rendered.png")
                                    try:
                                        import base64
                                        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                                        raw_png = base64.b64decode(item['data'])
                                        with open(out_img, "wb") as f_out:
                                            f_out.write(raw_png)
                                        latest_render_bytes = raw_png
                                        if c_idx >= num_contents - 2:
                                            saw_render_in_trailing_tools = True
                                        if callback_context and hasattr(callback_context, 'state'):
                                            callback_context.state['latest_render_path'] = out_img
                                            if agent_name == 'critic':
                                                callback_context.state['critic_rendered_this_iter'] = True
                                    except Exception:
                                        pass
                                    sanitized.append({
                                        'type': 'text',
                                        'text': f'[CAD drawing screenshot rendered and saved to {out_img}]'
                                    })
                                else:
                                    sanitized.append(item)
                            resp['content'] = sanitized
                        elif fn_name in ('render', 'get_screenshot'):
                            # Handle case where session history already stripped inline base64
                            if callback_context and hasattr(callback_context, 'state') and agent_name == 'critic':
                                callback_context.state['critic_rendered_this_iter'] = True
                    new_parts.append(part)
                content.parts = new_parts

        # Inject the latest rendered CADCraft PNG as a visual user message so Critic (or Draftsman/Refiner)
        # can actually inspect the drawing visually side-by-side with the original blueprint!
        should_show_render = saw_render_in_trailing_tools or (
            agent_name == 'critic'
            and callback_context
            and hasattr(callback_context, 'state')
            and bool(callback_context.state.get('critic_rendered_this_iter', False))
        )
        if should_show_render:
            if not latest_render_bytes and callback_context and hasattr(callback_context, 'state'):
                rp = callback_context.state.get('latest_render_path') or str(OUTPUT_DIR / f"{part_name}_rendered.png")
                if rp and os.path.isfile(rp):
                    try:
                        latest_render_bytes = Path(rp).read_bytes()
                    except Exception:
                        latest_render_bytes = None
            if latest_render_bytes:
                bounded_render = _prepare_image_bytes(latest_render_bytes, enhance_blueprint=False, max_dim=1400) or latest_render_bytes
                render_prompt = types.Part.from_text(
                    text=(
                        f"[RENDERED CADCRAFT OUTPUT IMAGE ({part_name}_rendered.png)] "
                        "Inspect this rendered CADCraft drawing carefully and compare its outer silhouette, "
                        "arc curvatures (concave vs convex), concentric circles, hole positions, and tangent joints "
                        "directly against the ORIGINAL technical drawing image uploaded earlier."
                    )
                )
                render_img_part = types.Part.from_bytes(data=bounded_render, mime_type='image/png')
                last_c = llm_request.contents[-1]
                last_has_fn_resp = any(getattr(p, 'function_response', None) for p in (getattr(last_c, 'parts', None) or []))
                if getattr(last_c, 'role', '') == 'user' and not last_has_fn_resp:
                    last_c.parts.extend([render_prompt, render_img_part])
                else:
                    llm_request.contents.append(types.Content(role='user', parts=[render_prompt, render_img_part]))

    return None

retry_policy = RetryConfig(max_attempts=1, initial_delay=1.0)
gen_config = types.GenerateContentConfig(
    max_output_tokens=4096,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(initial_delay=1.0, attempts=1)
    )
)


def python_tool(code: str) -> str:
    """Executes Python code for mathematical calculations, trigonometry, and geometric intersections.
    Includes `math`, `numpy` (as `np`), and all `geometry_tools` functions directly in scope.
    Prints formulas and calculated results to stdout.
    """
    import sys, io, traceback, math
    from sketchcad_agent import geometry_tools as _gt
    local_env = {
        "math": math,
        "fillet_two_lines": _gt.fillet_two_lines,
        "tangent_lines_two_circles": _gt.tangent_lines_two_circles,
        "tangent_from_point_to_circle": _gt.tangent_from_point_to_circle,
        "circle_circle_intersections": _gt.circle_circle_intersections,
        "circle_line_intersections": _gt.circle_line_intersections,
        "arc_angles": _gt.arc_angles,
        "line_line_intersection": _gt.line_line_intersection,
        "tangent_arc_line_circle": _gt.tangent_arc_line_circle,
        "tangent_arc_two_circles": _gt.tangent_arc_two_circles,
        "tangent_point_and_vector": _gt.tangent_point_and_vector,
    }
    try:
        import numpy as np
        local_env["np"] = np
        local_env["numpy"] = np
    except ImportError:
        pass

    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture
    try:
        try:
            compiled = compile(code.strip(), "<python_tool>", "eval")
            res = eval(compiled, local_env)
            if res is not None:
                print(res)
        except SyntaxError:
            compiled = compile(code, "<python_tool>", "exec")
            exec(compiled, local_env)
        output = stdout_capture.getvalue().strip()
        return output if output else "Executed successfully (no output)."
    except Exception as e:
        return f"Error executing Python code: {e}\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout


async def _init_pipeline_state(callback_context):
    """Resets per-run pipeline state cleanly before Analyst runs."""
    if callback_context and hasattr(callback_context, 'state'):
        callback_context.state.setdefault('part_name', 'mechanical_part')
        callback_context.state.setdefault('drawing_spec', '{}')
        callback_context.state.setdefault('cad_description', '')
        callback_context.state.setdefault('critic_report', '{}')
        callback_context.state['analyst_tool_calls'] = 0
        callback_context.state['critic_rendered_this_iter'] = False
        callback_context.state.pop('latest_render_path', None)
    return None


async def _reset_critic_render_flag(callback_context):
    """Marks that Draftsman/Refiner is editing the drawing so Critic must call render() fresh."""
    if callback_context and hasattr(callback_context, 'state'):
        callback_context.state['critic_rendered_this_iter'] = False
    return None


analyst_output_schema = None
critic_output_schema = None if MODEL.startswith('openai/') else CriticReport

analyst = LlmAgent(
    name='analyst',
    model=active_model,
    instruction=ANALYST_PROMPT,
    tools=[
        python_tool,
        fillet_two_lines,
        tangent_lines_two_circles,
        tangent_from_point_to_circle,
        circle_circle_intersections,
        circle_line_intersections,
        arc_angles,
        line_line_intersection,
        tangent_arc_line_circle,
        tangent_arc_two_circles,
        tangent_point_and_vector,
    ],
    output_schema=None,
    output_key='cad_description',
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_agent_callback=_init_pipeline_state,
    before_model_callback=_pace_and_preprocess,
    after_model_callback=_sanitize_analyst_output
)

if CAD_BACKEND == 'cadcraft':
    active_cad = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=CADCRAFT_CLI_CMD,
                args=['mcp']
            ),
            timeout=60
        ),
        tool_filter=[
            'new_drawing', 'command_line', 'script', 'execute',
            'render', 'save', 'inspect_drawing', 'query_entities',
            'find_text', 'open', 'cancel', 'list_commands'
        ]
    )
    draftsman_instruction = DRAFTSMAN_CADCRAFT_PROMPT
    critic_instruction = CRITIC_CADCRAFT_PROMPT
    refiner_instruction = REFINER_CADCRAFT_PROMPT
else:
    active_cad = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=FREECAD_MCP_CMD,
                env=dict(os.environ, FREECAD_MODE='xmlrpc')
            ),
            timeout=60
        ),
        tool_filter=[
            'create_document', 'create_partdesign_body', 'create_sketch',
            'add_sketch_line', 'add_sketch_circle', 'add_sketch_arc',
            'add_sketch_rectangle', 'add_sketch_point', 'recompute_document',
            'inspect_object', 'list_objects', 'undo', 'save_document',
            'fit_all', 'get_screenshot', 'execute_python'
        ]
    )
    draftsman_instruction = DRAFTSMAN_PROMPT
    critic_instruction = CRITIC_PROMPT
    refiner_instruction = REFINER_PROMPT

# Keep alias references for backward compatibility
freecad = active_cad
cadcraft = active_cad

async def read_entity(id: str = "") -> dict:
    """Reads an entity by spec id or handle and returns its exact geometric coordinates, angles, and dimensions."""
    global _entity_id_map, _last_created_handle
    try:
        res = await active_cad._execute_with_session(lambda s: s.call_tool('query_entities', {}), 'Error in query_entities')
        data = json.loads(res.content[0].text if res.content else '{}')
        ents = data.get('entities', [])
        if not ents:
            return {'error': 'No entities in drawing', 'status': 'not_found'}

        target_handle = _entity_id_map.get(id, id)
        target = None
        if target_handle:
            for e in ents:
                if str(e.get('handle')) == str(target_handle):
                    target = e
                    break
        if not target:
            target = ents[-1]
            if id:
                _entity_id_map[id] = str(target.get('handle'))

        g = target.get('geometry', {})
        t = target.get('type', '').lower()
        h = str(target.get('handle'))
        _last_created_handle = h

        if t == 'line':
            p1 = [round(g['a']['x'], 4), round(g['a']['y'], 4)]
            p2 = [round(g['b']['x'], 4), round(g['b']['y'], 4)]
            length = round(math.hypot(p2[0] - p1[0], p2[1] - p1[1]), 4)
            return {'id': id or h, 'handle': h, 'type': 'line', 'p1': p1, 'p2': p2, 'endpoints': [p1, p2], 'length': length}
        elif t == 'arc':
            cx, cy = round(g['center']['x'], 4), round(g['center']['y'], 4)
            r = round(g['radius'], 4)
            s_deg = round(math.degrees(g['start']) % 360, 4)
            e_deg = round(math.degrees(g['end']) % 360, 4)
            sweep = round((e_deg - s_deg) % 360, 4)
            if sweep == 0 and abs(g['end'] - g['start']) > 1e-4:
                sweep = 360.0
            p_start = [round(cx + r * math.cos(math.radians(s_deg)), 4), round(cy + r * math.sin(math.radians(s_deg)), 4)]
            p_end = [round(cx + r * math.cos(math.radians(e_deg)), 4), round(cy + r * math.sin(math.radians(e_deg)), 4)]
            m_deg = (s_deg + sweep / 2.0) % 360
            mid = [round(cx + r * math.cos(math.radians(m_deg)), 4), round(cy + r * math.sin(math.radians(m_deg)), 4)]
            return {'id': id or h, 'handle': h, 'type': 'arc', 'center': [cx, cy], 'radius': r, 'start_angle': s_deg, 'end_angle': e_deg, 'sweep': sweep, 'midpoint': mid, 'p_start': p_start, 'p_end': p_end}
        elif t == 'circle':
            cx, cy = round(g['center']['x'], 4), round(g['center']['y'], 4)
            r = round(g['radius'], 4)
            return {'id': id or h, 'handle': h, 'type': 'circle', 'center': [cx, cy], 'radius': r}
        return {'id': id or h, 'handle': h, 'type': t, 'geometry': g}
    except Exception as exc:
        return {'error': str(exc), 'status': 'failed'}

async def delete_entity(id: str = "") -> dict:
    """Deletes an entity by spec id or handle using CADCraft erase."""
    global _entity_id_map
    target_handle = _entity_id_map.get(id, id)
    if not target_handle and _last_created_handle:
        target_handle = _last_created_handle
    try:
        await active_cad._execute_with_session(
            lambda s: s.call_tool('execute', {'command': 'erase', 'params': {'handles': [str(target_handle)]}}),
            'Error in delete_entity'
        )
        if id in _entity_id_map:
            del _entity_id_map[id]
        return {'status': 'deleted', 'id': id, 'handle': str(target_handle)}
    except Exception as exc:
        return {'error': str(exc), 'status': 'failed'}

def _representative_point(ent: dict, toward: Optional[list[float]] = None) -> list[float]:
    """Computes a pick point lying cleanly on a line, arc, or circle entity."""
    etype = ent.get('type', '')
    if etype == 'circle' and ent.get('center') and ent.get('radius'):
        cx, cy = ent['center']
        r = float(ent['radius'])
        if toward:
            dx, dy = toward[0] - cx, toward[1] - cy
            d = math.hypot(dx, dy)
            if d > 1e-6:
                return [cx + r * dx / d, cy + r * dy / d]
        return [cx + r, cy]
    if etype == 'arc' and ent.get('midpoint'):
        return ent['midpoint']
    p1 = ent.get('p1')
    p2 = ent.get('p2')
    if p1 and p2:
        return [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0]
    return ent.get('center') or [0.0, 0.0]


async def add_fillet(
    entity1: str,
    entity2: str,
    radius: float,
    pick1: Optional[list[float]] = None,
    pick2: Optional[list[float]] = None,
) -> dict:
    """Applies a fillet between two entities (lines, arcs, or circles) with radius R and inserts the fillet arc."""
    global _entity_id_map
    try:
        e1 = await read_entity(entity1)
        e2 = await read_entity(entity2)
        if 'error' in e1 or 'error' in e2:
            return {'error': f'Entity not found: {e1.get("error") or e2.get("error")}', 'status': 'failed'}

        h1 = e1['handle']
        h2 = e2['handle']

        if e1.get('type') == 'line' and e2.get('type') == 'line':
            l1 = e1.get('length', 999.0)
            l2 = e2.get('length', 999.0)
            if radius > min(l1, l2) + 0.05:
                return {
                    'error': f'Radius {radius} is too large for neighbouring line segments (lengths {l1:.3f}, {l2:.3f}).',
                    'status': 'failed'
                }

        if pick1 is None or pick2 is None:
            pts1 = [p for p in (e1.get('endpoints') or [e1.get('p1'), e1.get('p2')]) if p]
            pts2 = [p for p in (e2.get('endpoints') or [e2.get('p1'), e2.get('p2')]) if p]
            if e1.get('type') == 'line' and e2.get('type') == 'line' and len(pts1) == 2 and len(pts2) == 2:
                best_dist = 999999.0
                c1, c2 = pts1[0], pts2[0]
                for p_a in pts1:
                    for p_b in pts2:
                        d = math.hypot(p_a[0] - p_b[0], p_a[1] - p_b[1])
                        if d < best_dist:
                            best_dist = d
                            c1, c2 = p_a, p_b
                o1 = pts1[1] if pts1[0] == c1 else pts1[0]
                o2 = pts2[1] if pts2[0] == c2 else pts2[0]
                pick1 = pick1 or [c1[0] + 0.1 * (o1[0] - c1[0]), c1[1] + 0.1 * (o1[1] - c1[1])]
                pick2 = pick2 or [c2[0] + 0.1 * (o2[0] - c2[0]), c2[1] + 0.1 * (o2[1] - c2[1])]
            else:
                ref2 = _representative_point(e2)
                ref1 = _representative_point(e1, toward=ref2)
                pick1 = pick1 or _representative_point(e1, toward=ref2)
                pick2 = pick2 or _representative_point(e2, toward=ref1)

        res = await active_cad._execute_with_session(
            lambda s: s.call_tool('execute', {'command': 'fillet', 'params': {'h1': h1, 'p1': pick1, 'h2': h2, 'p2': pick2, 'radius': radius, 'trim': True}}),
            'Error in fillet'
        )
        res_text = res.content[0].text if res.content else '{}'
        res_data = json.loads(res_text) if '{' in res_text else {}
        arc_h = res_data.get('arc')
        if not arc_h:
            return {'error': f'CADCraft fillet failed: {res_text}', 'status': 'failed'}

        arc_info = await read_entity(arc_h)
        return {'status': 'ok', 'arc_handle': arc_h, 'fillet_arc': arc_info, 'radius': radius}
    except Exception as exc:
        return {'error': str(exc), 'status': 'failed'}

GEOMETRY_TOOLS = [
    fillet_two_lines,
    tangent_lines_two_circles,
    tangent_from_point_to_circle,
    circle_circle_intersections,
    circle_line_intersections,
    arc_angles,
    line_line_intersection,
    trim_extend_lines,
    chamfer_two_lines,
    close_joint,
    trim_circle_or_arc,
    tangent_arc_line_circle,
    tangent_arc_two_circles,
    tangent_point_and_vector,
]

draftsman = LlmAgent(
    name='draftsman',
    model=active_model,
    instruction=draftsman_instruction,
    tools=[
        active_cad,
        python_tool,
        add_fillet,
        read_entity,
        delete_entity,
        open_in_cadcraft_gui,
        *GEOMETRY_TOOLS,
    ],
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_agent_callback=_reset_critic_render_flag,
    before_model_callback=_pace_and_preprocess,
    after_model_callback=_parse_and_inject_tool_calls
)

critic = LlmAgent(
    name='critic',
    model=active_model,
    instruction=critic_instruction,
    tools=[active_cad, open_in_cadcraft_gui, exit_loop],
    output_schema=critic_output_schema,
    output_key='critic_report',
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_model_callback=_pace_and_preprocess,
    after_model_callback=_sanitize_critic_output
)

refiner = LlmAgent(
    name='refiner',
    model=active_model,
    instruction=refiner_instruction,
    tools=[
        active_cad,
        python_tool,
        add_fillet,
        read_entity,
        delete_entity,
        open_in_cadcraft_gui,
        *GEOMETRY_TOOLS,
    ],
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_agent_callback=_reset_critic_render_flag,
    before_model_callback=_pace_and_preprocess,
    after_model_callback=_parse_and_inject_tool_calls
)


refinement_loop = LoopAgent(
    name='refinement_loop',
    sub_agents=[critic, refiner],
    max_iterations=4
)

root_agent = SequentialAgent(
    name='sketchcad',
    sub_agents=[analyst, draftsman, refinement_loop],
    before_agent_callback=_init_pipeline_state,
)
