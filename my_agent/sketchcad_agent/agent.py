import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional
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
)

load_dotenv()

OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY')
if OPENROUTER_KEY:
    os.environ['OPENROUTER_API_KEY'] = OPENROUTER_KEY

MODEL = os.getenv('SKETCHCAD_MODEL')
if not MODEL:
    if OPENROUTER_KEY:
        MODEL = 'openrouter/google/gemini-2.5-flash'
    else:
        MODEL = 'gemini-3.5-flash'
elif OPENROUTER_KEY and not MODEL.startswith(('openrouter/', 'gemini-', 'gemma-')):
    MODEL = f'openrouter/{MODEL}'

CAD_BACKEND = os.getenv('CAD_BACKEND', 'cadcraft').lower()
CADCRAFT_CLI_CMD = shutil.which('cadcraft-cli') or '/usr/bin/cadcraft-cli'
FREECAD_MCP_CMD = shutil.which('freecad-mcp') or str(Path(sys.executable).parent / 'freecad-mcp')

_cadcraft_gui_proc = None

def open_in_cadcraft_gui(file_path: str = "OUT.dxf") -> str:
    """Opens CADCraft desktop GUI application with the generated CAD file to showcase the output directly on screen.

    Args:
        file_path: Relative or absolute path to the DXF or DWG file (defaults to 'OUT.dxf').
    """
    global _cadcraft_gui_proc
    cadcraft_gui = shutil.which('cadcraft') or '/usr/bin/cadcraft'
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        return f"Error: CAD file not found at {abs_path}. Please save the drawing first using save(path='{file_path}')."

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

class DrawingSpec(BaseModel):
    units: Literal['mm']
    entities: list[Entity]
    relations: list[Relation]
    annotations: list[TextAnnotation] = []
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

ANALYST_PROMPT = """You are the Analyst of a 2D CAD pipeline. The user attached an image or
PDF of a drawing. Output ONLY the DrawingSpec.
Work in three passes:
1. TRANSCRIBE every number, unit and label you can see. Do not interpret yet.
2. STRUCTURE: list each line, circle and arc with coordinates in mm.
   Origin at the bottom-left of the main shape, y pointing up. Link each
   dimension to the entity it measures as a relation (distance or radius).
3. CHECK: parts must sum to totals, outlines must close, and every
   dimension must agree with the coordinates. Fix what fails.
Rules:
- Complete Profile Extraction: You must extract the ENTIRE profile shown in the drawing:
  1. Base line and corner fillets.
  2. Side profiles (vertical lines, reverse curves, slopes).
  3. Main outer circle or arc (e.g. R 0.55h).
  4. Protruding tabs, ears, or angled features (e.g. at 60 deg).
  5. Inner bore or circle (e.g. R 0.3h).
  6. Any internal cutouts, keyways, slots, or notches.
  Never output only 2 or 3 basic entities for a full mechanical drawing.
- Object Lines vs Dimension Lines: Only extract physical contour lines and features of the part.
  NEVER extract dimension lines, witness/extension lines, leader lines, or bounding box lines as entities.
- Parametric dimensions in terms of 'h': If dimensions are given as expressions of 'h' (e.g. 0.55h, 0.3h, 1.185h),
  set h = 100.0 mm and calculate exact numeric values in mm (e.g. 0.55*100 = 55.0 mm).
- DO NOT SIMPLIFY: Never omit features, slots, notches, cutouts, angles, or fillets.
- Dimensions are the truth. If a stated dimension disagrees with the
  picture, use the stated number and note it in assumptions.
- Convert to mm (1 cm = 10 mm, 1 inch = 25.4 mm, 1 ft = 304.8 mm).
- Add tangent, coincident, horizontal, vertical, perpendicular or equal
  relations only where the drawing clearly shows them. Log every guess
  in assumptions.
- Never guess a digit you cannot read: list it in unreadable and leave
  that dimension out.
- With several views, use only the main view and say so in assumptions.
- Spatial Contract (DeciCAD standard):
  * Units: strictly millimeters (mm) for lengths, degrees (deg) for angles.
  * World frame: Right-handed Cartesian (+X right, +Y up/back, +Z up).
  * Sketch frame: Local 2D right-handed plane coordinates.
  * Rotations: Follow standard right-hand rule (axis, angle).
- For rounded corners, tangent joins and notches, describe the INTENT.
  Draw lines to the sharp corner they would meet at. Add a relation of kind
  fillet (refs: the two line ids, value_mm: the radius). Give circles as
  center and radius. Add tangent relations between lines and circles. Do
  not compute arc endpoints or angles: the Draftsman has tools for that.
- Strict Origin Anchoring at (0.0, 0.0):
  * The origin (0.0, 0.0) MUST be fixed at the starting vertex of the main contour (bottom-left corner / base start).
  * Entity e1 MUST begin at p1 = [0.0, 0.0].
  * EVERY line endpoint, circle center, arc center, dimension offset, and text annotation position MUST be calculated and specified strictly in reference to this (0.0, 0.0) origin.
  * In assumptions, always explicitly include: "Origin (0.0, 0.0) is fixed at the base starting point; all coordinates are referenced from this origin."
- Text & Annotations: Detect and extract all text, callouts, notes, and labels
  (e.g. 'R30', 'R55', '50.0', '100', 'KEYWAY', 'SECTION A-A') into annotations:
  list of TextAnnotation objects with text, position [x, y] in mm, height in mm
  (default 3.0), and rotation_deg (default 0.0)."""

DRAFTSMAN_PROMPT = """You are the Draftsman. Draw the FIRST DRAFT of this DrawingSpec in FreeCAD on the 2D draft plane.
DrawingSpec: {drawing_spec}

Procedure:
1. create_document, then create_sketch on XY (in 2D Draft mode, do not create a PartDesign body).
2. Draw the actual physical part contours and cutouts (lines, arcs, circles):
   - Do NOT draw dimension extension lines or outer bounding boxes.
   - Batch additions efficiently using execute_python or tool calls.
   - Apply geometric constraints (coincident, horizontal, vertical, tangent, distance, radius).
3. Call recompute_document, fit_all, then call get_screenshot(view_angle='Top') to capture the first draft.
4. Export DXF with execute_python:
   import FreeCAD as App, importDXF
   importDXF.export([App.ActiveDocument.getObject('SKETCH_NAME')], 'OUT.dxf')
5. Conclude: Summarize what was drawn in this initial first draft.
Rules: never invent tool names. Coordinates come only from the spec. At most 40 tool calls.

DECICAD HARNESS PRINCIPLES (Semantic Decision Bottleneck & State Grounding):
- Spatial Contract: Right-handed Cartesian coordinates (+X right, +Y back/up, +Z up), lengths in mm, angles in deg.
- State Grounding: All actions execute against the persistent FreeCAD document. Never clear or recreate from scratch when recovering; build incrementally upon existing stable entity references.
- Two-Tier Observation: Read the immediate execution feedback of each tool call. When solver or geometry state is uncertain, use active hierarchical inspection (list_objects, inspect_object) rather than guessing.
- Structured Error Recovery:
  * InvalidInput: Check dimensions, use geometry calculators for exact numbers (4+ decimals).
  * Conflict / Overconstrained: Redundant constraints detected. A Tangent constraint already enforces coincidence; NEVER add duplicate Coincident constraints to the same tangent vertex.
  * GeometryFailure: Degenerate or open loop. Verify each line endpoint matches the next arc/line start point.
  * OutcomeUnknown: Call get_screenshot or inspect_object to observe state before proceeding.

CURVES AND CORNERS. Never estimate arc endpoints or angles yourself.
- Rounded corner between two lines: call fillet_two_lines. Draw each line
  only up to the tangent point it returns. Draw the arc with the returned
  center, radius, start_deg and end_deg.
- Line tangent between two circles: tangent_lines_two_circles.
- Line from a point tangent to a circle: tangent_from_point_to_circle.
- Circle cut by another circle (notch): circle_circle_intersections, then
  arc_angles with a point on the side to KEEP. Draw only the kept arcs.
  Never use trim.
- Arcs run counter-clockwise from start_deg to end_deg.
- Use exact numbers from the tools, with at least 4 decimals.
For a tangent joint between an arc end and a line end use ONE constraint:
   sk.addConstraint(Sketcher.Constraint('Tangent', ARC_ID, ARC_POINT, LINE_ID, LINE_POINT))
It already includes the coincidence. Never add a separate coincident for
the same joint: that causes redundant-constraint solver failures.
Add Radius constraints on arcs after the tangents. If the solver fails,
undo, report it, and do not retry with different numbers.

FLAT LINES MEETING A CIRCLE (seal and profile outlines). When a circle is
merged into a flat line, never draw a half circle and never use trim.
1. Call circle_line_intersections for each line against the circle and
   choose the point on the drawn part of the outline.
2. Call arc_angles with those two points and via_point = a point on the
   circle on the side the outline keeps (for example the top). Draw only
   that arc.
3. Start or end each line AT the intersection point, so the arc and the
   line share an exact endpoint.
4. Build the outline as one closed loop in order. Check that each end
   point equals the next start point before adding any constraint."""

CRITIC_PROMPT = """You are the Visual Critic / Inspector of the CAD pipeline.
Compare the current FreeCAD 2D drawing screenshot directly against the ORIGINAL TECHNICAL DRAWING IMAGE uploaded by the user!
Do NOT assume {drawing_spec} is complete — {drawing_spec} may be missing critical entities.

Your Goal: Strictly inspect the visual geometry in FreeCAD against the user's drawing image. Target accuracy is 98% and above.

Procedure:
1. Inspect the FreeCAD drawing: call fit_all, then call get_screenshot(view_angle='Top') to inspect the visual rendering.
2. Conduct a strict visual audit across 5 mandatory checks:
   - Check A (Base & Fillets): Base line (100 mm) and bottom corner fillets (R=3 mm).
   - Check B (Left Profile): Lower vertical edge, left reverse bulge (R=8.65 mm), and left tangent transition.
   - Check C (Main Outer Head): Top circular arc (R=55 mm, center 50,45).
   - Check D (Right Protrusion & Slant): Right ear/protrusion at 60 deg angle (length 50 mm, width 12.5 mm, R=3 mm fillets) and slant down at 140 deg to base.
   - Check E (Inner Features): Central bore hole (R=30 mm) and keyway slot/cutout at 20 deg angle.
3. Strict Scoring Rules:
   - If Check D (Right Protrusion / Slant) or Check E (Slot cutout) are MISSING from the FreeCAD drawing:
     ACCURACY SCORE CANNOT EXCEED 65.0%.
     IT IS STRICTLY FORBIDDEN TO CALL exit_loop() WHEN ANY OF CHECKS A, B, C, D, E ARE MISSING.
   - Score >= 98.0% ONLY IF ALL 5 CHECKS (A, B, C, D, E) are completely drawn, topologically connected, and match the dimensions.
4. If accuracy_score >= 98.0:
   - Call the exit_loop tool to complete the loop!
   - Set is_accurate to True and summarize that the drawing meets the quality target (>= 98%).
5. If accuracy_score < 98.0:
   - Do NOT call exit_loop.
   - Set is_accurate to False.
   - In errors: detail each missing check with structured issue_type ('missing_feature', 'open_contour', 'geometry_conflict', 'dimension_mismatch', or 'invalid_state').
   - In rectification_commands: provide exact FreeCAD Python commands for the Refiner to execute incrementally in the active sketch (e.g. adding missing lines, arcs, or correcting constraints).
"""

REFINER_PROMPT = """You are the CAD Refiner. Your job is to rectify errors reported by the Critic and update the FreeCAD drawing.
Critic Report:
{critic_report}

DrawingSpec:
{drawing_spec}

Procedure:
1. Review the errors and rectification_commands from the Critic Report.
2. Apply the corrections in FreeCAD using execute_python:
   - Execute the Critic's rectification commands to add the missing geometry (arcs, lines, circles, slots) into the active sketch.
   - Follow DeciCAD state grounding: modify the persistent active sketch incrementally; never recreate the sketch from scratch.
   - Ensure new geometry endpoints connect properly to form closed loops.
3. Call recompute_document, fit_all, and call get_screenshot(view_angle='Top') to capture the updated drawing.
4. Summarize what was added/fixed so the Critic can re-evaluate on the next iteration of the loop.
Rules: never invent tool names. Follow the Critic's rectification commands strictly."""

DRAFTSMAN_CADCRAFT_PROMPT = """You are the Draftsman. Draw the FIRST DRAFT of this DrawingSpec in CADCraft (2D CAD).
DrawingSpec: {drawing_spec}

Procedure:
1. Initialize drawing: call new_drawing(metric=True).
2. STRICT (0.0, 0.0) ORIGIN ANCHORING (MANDATORY):
   - You MUST begin drawing starting from the origin point (0.0, 0.0).
   - The first entity command MUST begin at exactly (0.0, 0.0), e.g. command_line(text="LINE 0.0,0.0 <x2>,<y2> ").
   - EVERY subsequent line, arc, circle, cutout, slot, dimension, and text annotation MUST be drawn with coordinates referenced strictly relative to that (0.0, 0.0) origin point.
   - Build a continuous topological chain starting from (0.0, 0.0) around the part geometry so that every single coordinate is referenced back to (0.0, 0.0).
   - Do NOT draw outer bounding boxes.
   - Use command_line or script with standard CAD commands:
     * Lines: command_line(text="LINE x1,y1 x2,y2 ")
     * Circles: command_line(text="CIRCLE cx,cy radius ")
     * Arcs: command_line(text="ARC start_x,start_y via_x,via_y end_x,end_y ")
     * Batch multiple commands at once using script(text="...").
   - Use the geometry calculator tools (fillet_two_lines, circle_line_intersections, arc_angles) for exact numbers!
   - Apply geometric and dimensional constraints using execute:
     * Tangent: execute(command="gctangent", params=...)
     * Coincident: execute(command="gccoincedent", params=...)
     * Horizontal: execute(command="gchorizontal", params=...)
     * Vertical: execute(command="gcvertical", params=...)
     * Radius: execute(command="dcradius", params=...)

3. SHOW DIMENSIONS FOR EACH LINE & FEATURE (MANDATORY):
   Draw dimensions for the lines, arcs, circles, and features so dimensions are shown while viewing:
   - Linear Dimensions: execute(command="dimlinear", params={"p1": [x1, y1], "p2": [x2, y2], "at": [at_x, at_y]})
     where [at_x, at_y] is offset 5 to 10 mm away from the line outside the contour.
   - Radial / Diameter Dimensions: execute(command="dimradius", params={"center": [cx, cy], "point": [px, py]})
     or command_line(text="DIMRADIUS handle at ") for circles, arcs, and fillet curves.
   - Aligned Dimensions: execute(command="dimaligned", params={"p1": [x1, y1], "p2": [x2, y2], "at": [at_x, at_y]})
     for slanted/angled edges.

4. DRAW TEXT AND LABELS (MANDATORY):
   If there are text annotations in drawing_spec.annotations:
   - For every text item in drawing_spec.annotations, render it using CADCraft vector font text:
     execute(command="text", params={"at": [x, y], "text": "...", "height": height or 3.0, "rotation": rotation_deg or 0.0})
     or command_line(text="TEXT x,y height rotation text ")
     This replaces detected image text with crisp CAD fonts in the exact specified position referenced from (0.0, 0.0)!

5. Inspect and verify: call inspect_drawing(entities=True) to check entity counts and extents.
6. Render screenshot: call render(fit=True, path="OUT.png") to capture the visual rendering for the Critic.
7. Export DXF: call save(path="OUT.dxf").
8. Showcase in CADCraft GUI: call open_in_cadcraft_gui(file_path="OUT.dxf") so the CADCraft desktop application opens and showcases the final drawing directly on the user's screen!
9. Conclude: Summarize what was drawn, dimensioned, annotated with text, and showcased in CADCraft.
Rules: never invent tool names. Coordinates come only from the spec. At most 40 tool calls.

DECICAD HARNESS PRINCIPLES (Semantic Decision Bottleneck & State Grounding):
- Spatial Contract: Right-handed Cartesian coordinates (+X right, +Y back/up, +Z up), lengths in mm, angles in deg, origin anchored at (0.0, 0.0).
- State Grounding: All actions execute against the active CADCraft drawing. Build incrementally; never restart from scratch when recovering.
- Two-Tier Observation: Read execution feedback of each tool call. Check inspect_drawing or query_entities when state is uncertain.
- Structured Error Recovery:
  * InvalidInput: Check dimensions, use geometry calculators for exact numbers (4+ decimals).
  * Conflict: Do not overconstrain. Tangent joints naturally ensure continuity.
  * GeometryFailure: Degenerate or open loop. Ensure entity endpoints meet exactly.

CURVES AND CORNERS. Never estimate arc endpoints or angles yourself.
- Rounded corner between two lines: call fillet_two_lines. Draw each line
  only up to the tangent point it returns. Draw the arc with the returned
  center, radius, start_deg and end_deg.
- Line tangent between two circles: tangent_lines_two_circles.
- Line from a point tangent to a circle: tangent_from_point_to_circle.
- Circle cut by another circle (notch): circle_circle_intersections, then
  arc_angles with a point on the side to KEEP. Draw only the kept arcs.
  Never use trim.
- Arcs run counter-clockwise from start_deg to end_deg.
- Use exact numbers from the tools, with at least 4 decimals.

FLAT LINES MEETING A CIRCLE (seal and profile outlines). When a circle is
merged into a flat line, never draw a half circle and never use trim.
1. Call circle_line_intersections for each line against the circle and
   choose the point on the drawn part of the outline.
2. Call arc_angles with those two points and via_point = a point on the
   circle on the side the outline keeps (for example the top). Draw only
   that arc.
3. Start or end each line AT the intersection point, so the arc and the
   line share an exact endpoint.
4. Build the outline as one closed loop in order. Check that each end
   point equals the next start point before adding any constraint."""

CRITIC_CADCRAFT_PROMPT = """You are the Visual Critic / Inspector of the CAD pipeline.
Compare the current CADCraft 2D drawing screenshot directly against the ORIGINAL TECHNICAL DRAWING IMAGE uploaded by the user!
Do NOT assume {drawing_spec} is complete — {drawing_spec} may be missing critical entities.

Your Goal: Strictly inspect the visual geometry, dimensions, text annotations, and origin grounding in CADCraft against the user's drawing image. Target accuracy is 98% and above.

Procedure:
1. Inspect the CADCraft drawing: call render(fit=True, path="OUT.png") to inspect the visual rendering.
2. Conduct a strict visual audit across mandatory checks:
   - Check 0 (Origin Grounding): Drawing MUST start at origin (0.0, 0.0) with all coordinates referenced from (0.0, 0.0).
   - Check A (Base & Fillets): Base line (100 mm) starting at (0.0, 0.0) and corner fillets (R=3 mm).
   - Check B (Left Profile): Lower vertical edge, left reverse bulge (R=8.65 mm), and left tangent transition.
   - Check C (Main Outer Head): Top circular arc (R=55 mm, center 50,45).
   - Check D (Right Protrusion & Slant): Right ear/protrusion at 60 deg angle (length 50 mm, width 12.5 mm, R=3 mm fillets) and slant down at 140 deg to base.
   - Check E (Inner Features & Annotations): Central bore hole (R=30 mm), keyway slot at 20 deg, visible line dimensions, and text annotations.
3. Strict Scoring Rules:
   - If Check 0 (Origin at 0.0, 0.0), Check D (Right Protrusion / Slant) or Check E (Slot cutout) are MISSING from the CADCraft drawing:
     ACCURACY SCORE CANNOT EXCEED 65.0%.
     IT IS STRICTLY FORBIDDEN TO CALL exit_loop() WHEN ANY OF CHECKS 0, A, B, C, D, E ARE MISSING.
   - Score >= 98.0% ONLY IF ALL CHECKS (0, A, B, C, D, E) are completely drawn, topologically connected, dimensioned, anchored to (0.0, 0.0), and match the technical drawing.
4. If accuracy_score >= 98.0:
   - Call open_in_cadcraft_gui(file_path="OUT.dxf") to ensure the final drawing is showcased on the user's screen in CADCraft GUI!
   - Call the exit_loop tool to complete the loop!
   - Set is_accurate to True and summarize that the drawing meets the quality target (>= 98%).
5. If accuracy_score < 98.0:
   - Do NOT call exit_loop.
   - Set is_accurate to False.
   - In errors: detail each missing check with structured issue_type ('missing_feature', 'open_contour', 'geometry_conflict', 'dimension_mismatch', or 'invalid_state').
   - In rectification_commands: provide exact CADCraft commands (command_line or execute) for the Refiner to execute (e.g. adding missing lines, arcs, dimensions, or text anchored to origin).
"""

REFINER_CADCRAFT_PROMPT = """You are the CAD Refiner. Your job is to rectify errors reported by the Critic and update the CADCraft drawing.
Critic Report:
{critic_report}

DrawingSpec:
{drawing_spec}

Procedure:
1. Review the errors and rectification_commands from the Critic Report.
2. Apply the corrections in CADCraft:
   - Execute the Critic's rectification commands using command_line or execute to add the missing geometry (arcs, lines, circles, slots, dimensions, text) into the active drawing.
   - Follow DeciCAD state grounding: modify the active drawing incrementally; never recreate the drawing from scratch.
   - Maintain strict origin grounding: ensure all geometry is positioned relative to the (0.0, 0.0) origin.
   - Ensure new geometry endpoints connect properly to form closed loops.
3. Call save(path="OUT.dxf").
4. Call open_in_cadcraft_gui(file_path="OUT.dxf") to update the live CADCraft GUI window on the screen.
5. Call render(fit=True, path="OUT.png") to capture the updated drawing.
6. Summarize what was added/fixed so the Critic can re-evaluate on the next iteration of the loop.
Rules: never invent tool names. Follow the Critic's rectification commands strictly."""

import asyncio
import time
from google.genai import types
from google.adk.workflow._retry_config import RetryConfig

_last_call_time = 0.0
PACE_SECONDS = float(os.getenv('SKETCHCAD_PACE_SECONDS', '0.0'))

async def _pace_free_tier(callback_context, llm_request):
    """Pace API requests if configured."""
    global _last_call_time
    if PACE_SECONDS > 0:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < PACE_SECONDS:
            await asyncio.sleep(PACE_SECONDS - elapsed)
        _last_call_time = time.time()
    return None

retry_policy = RetryConfig(max_attempts=5, initial_delay=2.0)
gen_config = types.GenerateContentConfig(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(initial_delay=2.0, attempts=5)
    )
)

analyst = LlmAgent(
    name='analyst',
    model=MODEL,
    instruction=ANALYST_PROMPT,
    output_schema=DrawingSpec,
    output_key='drawing_spec',
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_model_callback=_pace_free_tier
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

draftsman = LlmAgent(
    name='draftsman',
    model=MODEL,
    instruction=draftsman_instruction,
    tools=[
        active_cad,
        open_in_cadcraft_gui,
        fillet_two_lines,
        tangent_lines_two_circles,
        tangent_from_point_to_circle,
        circle_circle_intersections,
        circle_line_intersections,
        arc_angles,
    ],
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_model_callback=_pace_free_tier
)

critic = LlmAgent(
    name='critic',
    model=MODEL,
    instruction=critic_instruction,
    tools=[active_cad, open_in_cadcraft_gui, exit_loop],
    output_schema=CriticReport,
    output_key='critic_report',
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_model_callback=_pace_free_tier
)

refiner = LlmAgent(
    name='refiner',
    model=MODEL,
    instruction=refiner_instruction,
    tools=[active_cad, open_in_cadcraft_gui],
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_model_callback=_pace_free_tier
)

refinement_loop = LoopAgent(
    name='refinement_loop',
    sub_agents=[critic, refiner],
    max_iterations=4
)

root_agent = SequentialAgent(
    name='sketchcad',
    sub_agents=[analyst, draftsman, refinement_loop]
)
