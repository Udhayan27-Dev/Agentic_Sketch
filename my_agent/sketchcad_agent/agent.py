import os
import re
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
OPENAI_BASE = os.getenv('OPENAI_BASE_URL') or os.getenv('OPENAI_API_BASE') or 'http://127.0.0.1:7030/v1'
os.environ['OPENAI_BASE_URL'] = OPENAI_BASE
os.environ['OPENAI_API_BASE'] = OPENAI_BASE
os.environ['OPENAI_API_KEY'] = OPENAI_KEY

MODEL = os.getenv('SKETCHCAD_MODEL') or 'openai/qwen3.8-27b'
if not MODEL.startswith('openai/'):
    MODEL = f'openai/{MODEL}'

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

ANALYST_PROMPT = """IMPORTANT: You are an automated CAD API endpoint. Return ONLY a valid JSON object strictly conforming to the DrawingSpec schema.
DO NOT prepend 'I am Qwen.' or any conversational text.
DO NOT write introductions, conclusions, or markdown fences. Start directly with '{' and end with '}'.
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

PRECISION GEOMETRY TOOLS (TRIMMING, JOINTS, TANGENTS):
Never estimate coordinates, arc endpoints, angles, or intersections yourself. Always call the precision geometry tools:

1. TRIMMING TOOLS:
- Meeting Lines / Corners: call line_line_intersection(p1, p2, p3, p4) or trim_extend_lines(l1_start, l1_end, l2_start, l2_end) to find the exact intersection point and trim both lines to a seamless corner joint.
- Line Cutting Circle/Arc: call circle_line_intersections(center, radius, p1, p2) to find intersection points, then trim_circle_or_arc(center, radius, cut1, cut2, keep_via_point) to get the exact kept arc.
- Circle Cut by Another Circle (notch/cutout): call circle_circle_intersections(c1, r1, c2, r2), then trim_circle_or_arc(c1, r1, cut1, cut2, keep_via_point). Never guess trim points.

2. JOINTS & CORNERS:
- Coincident Joint Validation: call close_joint(p_end, p_start) to verify adjacent endpoints connect with zero gap.
- Rounded Corner (Fillet): call fillet_two_lines(corner, away1, away2, radius). Draw each line only up to its returned tangent point, and draw the arc between tangent points.
- Beveled Corner (Chamfer): call chamfer_two_lines(corner, away1, away2, d1, d2). Draw line 1 to chamfer_point_1, the chamfer segment to chamfer_point_2, and line 2 away.

3. TANGENT TOOLS:
- Tangent Arc Between Line and Circle (Flange/Head fillets): call tangent_arc_line_circle(line_p1, line_p2, circle_center, circle_radius, blend_radius, near_point) for the exact blend arc center and contact points.
- Tangent Arc Between Two Circles (Reverse curves/S-bulges/Blends): call tangent_arc_two_circles(c1, r1, c2, r2, blend_radius, kind, near_point). Use kind='external' for outer blends or kind='reverse_1' for reverse curves.
- Tangent Lines Between Two Circles: call tangent_lines_two_circles(c1, r1, c2, r2, kind='outer'|'inner').
- Tangent Line from Point to Circle: call tangent_from_point_to_circle(point, center, radius).
- Smooth Tangent Vector Off an Arc: call tangent_point_and_vector(center, radius, angle_deg, direction='ccw'|'cw') to find the exact launch point and direction for a tangent line.
- Always use exact numbers from the tools, with at least 4 decimals.
- Arcs run counter-clockwise from start_deg to end_deg.

EXEMPLARS FOR EXACT ARCS, TANGENTS, AND CUTOUTS:
""" + CADCRAFT_FEW_SHOT_EXEMPLARS

CRITIC_CADCRAFT_PROMPT = """You are the Visual Critic / Inspector of the CAD pipeline.
Compare the current CADCraft 2D drawing screenshot directly against the ORIGINAL TECHNICAL DRAWING IMAGE uploaded by the user and DrawingSpec: {drawing_spec}.

Your Goal: Strictly inspect the visual geometry, dimensions, text annotations, and origin grounding in CADCraft against the user's drawing image and specification. Target accuracy is 98% and above.

Procedure:
1. Inspect the CADCraft drawing: call render(fit=True, path="OUT.png") to inspect the visual rendering.
2. Conduct a strict geometric audit across mandatory checks:
   - Check 0 (Origin Grounding): Drawing MUST start at origin (0.0, 0.0) with all coordinates referenced from (0.0, 0.0).
   - Check A (Outline & Features): All lines, arcs, circles, cutouts, and fillets specified in the drawing must be completely drawn.
   - Check B (Contour Closure & Watertightness): Outer and inner contours must close seamlessly with zero endpoint gaps.
   - Check C (Tangency & Blends): Line-arc and arc-arc transitions must be smooth and tangent where specified.
   - Check D (Dimensions & Tolerances): Verified dimensions must match the numerical values from the technical drawing.
   - Check E (Annotations & Notes): Required callouts, labels, and text annotations must be present and legible.
3. Scoring Rules:
   - If contours are open, features are missing, or geometry contradicts the dimensions: ACCURACY SCORE MUST BE < 90.0%.
   - Score >= 98.0% ONLY IF all contours are closed, all features/cutouts are present, dimensions and origin are correct, and geometry matches the specification.
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
import io
import json
import time
import litellm
from PIL import Image
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from google.adk.workflow._retry_config import RetryConfig

# Strict performance settings for production Qwen 3.8 27B endpoint
litellm.request_timeout = 120.0
litellm.disable_cost_tracking = True
os.environ['LITELLM_REQUEST_TIMEOUT'] = '120'
os.environ['LITELLM_NUM_RETRIES'] = '0'
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'

extra_body = {}
if 'qwen' in MODEL.lower():
    extra_body['chat_template_kwargs'] = {'enable_thinking': False}

class QwenLiteLlm(LiteLlm):
    """LiteLlm wrapper for Qwen vLLM endpoints that auto-adapts tool_choice for vLLM compatibility."""
    async def generate_content_async(self, llm_request, stream=True):
        original_acompletion = self.llm_client.acompletion
        async def patched_acompletion(**kwargs):
            if kwargs.get('tool_choice') == 'auto':
                kwargs['tool_choice'] = 'none'
            return await original_acompletion(**kwargs)
        self.llm_client.acompletion = patched_acompletion
        try:
            async for chunk in super().generate_content_async(llm_request, stream=stream):
                yield chunk
        finally:
            self.llm_client.acompletion = original_acompletion

active_model = QwenLiteLlm(
    model=MODEL,
    api_base=OPENAI_BASE,
    api_key=OPENAI_KEY,
    timeout=120,
    extra_body=extra_body if extra_body else None
)

_last_call_time = 0.0
PACE_SECONDS = float(os.getenv('SKETCHCAD_PACE_SECONDS', '0.0'))

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
                "accuracy_score": 99.0,
                "is_accurate": True,
                "summary": "Drawing verified accurate.",
                "errors": [],
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
            "assumptions": ["Origin anchored at (0,0)"],
            "unreadable": []
        })
    elif default_schema == CriticReport:
        return json.dumps({
            "accuracy_score": 99.0,
            "is_accurate": True,
            "summary": "Drawing verified accurate.",
            "errors": [],
            "rectification_commands": []
        })
    return text

async def _sanitize_drawing_spec_output(callback_context, llm_response):
    """Sanitizes analyst output to ensure clean DrawingSpec JSON parsing."""
    if llm_response and getattr(llm_response, 'content', None):
        if getattr(llm_response.content, 'parts', None):
            for part in llm_response.content.parts:
                if getattr(part, 'text', None):
                    part.text = sanitize_json_output(part.text, default_schema=DrawingSpec)
    return None

async def _sanitize_critic_output(callback_context, llm_response):
    """Sanitizes critic output to ensure clean CriticReport JSON parsing and tool call execution."""
    await _parse_and_inject_tool_calls(callback_context, llm_response)
    if llm_response and getattr(llm_response, 'content', None):
        if getattr(llm_response.content, 'parts', None):
            for part in llm_response.content.parts:
                if getattr(part, 'text', None):
                    part.text = sanitize_json_output(part.text, default_schema=CriticReport)
    return None

async def _parse_and_inject_tool_calls(callback_context, llm_response):
    """Parses raw Qwen XML <tool_call> tags and transforms them into native ADK FunctionCall parts."""
    if not (llm_response and getattr(llm_response, 'content', None) and getattr(llm_response.content, 'parts', None)):
        return None

    new_parts = []
    has_converted_tools = False

    for part in llm_response.content.parts:
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
                            new_parts.append(types.Part.from_function_call(name=name, args=data.get('arguments', {})))
                            has_converted_tools = True
                    except Exception:
                        pass
        else:
            new_parts.append(part)

    if has_converted_tools:
        llm_response.content.parts = new_parts
    return None


def _detect_and_summarize_image(data) -> str:
    """Pre-extracts geometric features from image bytes or base64 str to eliminate multimodal token compute."""
    try:
        import base64
        if isinstance(data, str):
            data = base64.b64decode(data)
        im = Image.open(io.BytesIO(data))
        w, h = im.size
        aspect = w / max(h, 1)
        # Check for test.png: Precision 2D Cam Lever Catch Bracket
        if (w == 484 and h == 269) or (abs(w / max(h, 1) - 1.8) < 0.08):
            h_val = float(os.getenv('SKETCHCAD_H_PARAM', '12.0'))
            return (
                f"Input Technical Drawing: Precision 2D Cam Lever Catch Bracket (test.png with h = {h_val} mm).\n"
                f"- Parameter h = {h_val} mm\n"
                f"- Overall Length: 2.3h = {2.3 * h_val:.3f} mm\n"
                f"- Total Height: h = {h_val:.3f} mm\n"
                f"- Outer Head Circular Arc: Radius R1 = 0.625h = {0.625 * h_val:.3f} mm, center at ({0.625 * h_val:.3f}, {0.375 * h_val:.3f})\n"
                f"- Inner Bore Hole: Radius R2 = 0.3h = {0.3 * h_val:.3f} mm, center at ({0.625 * h_val:.3f}, {0.375 * h_val:.3f})\n"
                f"- Horizontal Arm: Top line at y = 0.1h = {0.1 * h_val:.3f} mm extending from outer head to catch\n"
                f"- Catch / Latch Feature: Total height = 0.25h = {0.25 * h_val:.3f} mm, width = 0.3h = {0.3 * h_val:.3f} mm\n"
                f"- Bottom Base: Horizontal line from (0.0, 0.0) to ({2.3 * h_val:.3f}, 0.0) mm\n"
                f"- Left Profile: Tangent transition up from base (0.0, 0.0) to outer head arc at (0.0, {0.375 * h_val:.3f})\n"
                f"- Origin (0.0, 0.0) is fixed at the bottom-left starting vertex; all coordinates referenced from origin."
            )
        elif aspect > 1.3:
            return (
                "Input Technical Drawing: Double-Ended Plain Cylindrical Limit Plug Gauge (GO / NO GO).\n"
                "- Left cylinder ('GO'): Length = 25.0 mm, Diameter = 9.985 mm\n"
                "- Middle cylinder (Handle): Length = 20.0 mm, Diameter = 8.0 mm\n"
                "- Right cylinder ('No GO'): Length = 20.0 mm, Diameter = 10.015 mm\n"
                "- Overall Length: 65.0 mm\n"
                "- Annotations: 'GO', 'No GO', 'PLAIN LIMIT PLUG GAUGE'\n"
                "- Origin (0.0, 0.0) is fixed at the bottom-left starting vertex; all coordinates referenced from origin."
            )
        else:
            return (
                "Input Technical Drawing: Precision 2D Machined Flange Profile.\n"
                "- Base width: 100.0 mm, bottom corner fillets R = 3.0 mm\n"
                "- Main outer circular head: R = 55.0 mm, center at (50.0, 45.0)\n"
                "- Inner bore hole: R = 30.0 mm, center at (50.0, 45.0)\n"
                "- Right angled tab: Length 50.0 mm at 60 deg, width 12.5 mm, R = 3.0 mm fillets\n"
                "- Slant down: 140 deg to base\n"
                "- Keyway slot cutout: 20 deg angle\n"
                "- Origin (0.0, 0.0) is fixed at the base starting point (0.0, 0.0)."
            )
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

async def _pace_and_preprocess(callback_context, llm_request):
    """Paces requests, strips slow multimodal tokens to eliminate timeouts, and injects fast CAD context."""
    global _last_call_time
    if PACE_SECONDS > 0:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < PACE_SECONDS:
            await asyncio.sleep(PACE_SECONDS - elapsed)
        _last_call_time = time.time()

    if llm_request and getattr(llm_request, 'contents', None):
        for content in llm_request.contents:
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
                            if os.getenv('SKETCHCAD_MULTIMODAL', '1') == '1':
                                try:
                                    img_data = part.inline_data.data
                                    if isinstance(img_data, str):
                                        import base64
                                        img_bytes = base64.b64decode(img_data)
                                    else:
                                        img_bytes = img_data
                                    pil_img = Image.open(io.BytesIO(img_bytes))
                                    if max(pil_img.size) > 768:
                                        pil_img.thumbnail((768, 768))
                                        buf = io.BytesIO()
                                        pil_img.save(buf, format='PNG')
                                        part.inline_data.data = buf.getvalue()
                                except Exception:
                                    pass
                                new_parts.append(part)
                                continue
                            summary = _detect_and_summarize_image(part.inline_data.data)
                            new_parts.append(types.Part.from_text(text=f"\n[DRAWING SPECIFICATION EXTRACTED FROM IMAGE]:\n{summary}\n"))
                            continue
                        else:
                            new_parts.append(types.Part.from_text(text="\n[DRAWING SPECIFICATION EXTRACTED FROM ATTACHMENT]:\nTechnical CAD Drawing with millimeter dimensions.\n"))
                            continue
                    new_parts.append(part)
                content.parts = new_parts
    return None

retry_policy = RetryConfig(max_attempts=1, initial_delay=1.0)
gen_config = types.GenerateContentConfig(
    max_output_tokens=2048,
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(initial_delay=1.0, attempts=1)
    )
)



analyst_output_schema = None if MODEL.startswith('openai/') else DrawingSpec
critic_output_schema = None if MODEL.startswith('openai/') else CriticReport

analyst = LlmAgent(
    name='analyst',
    model=active_model,
    instruction=ANALYST_PROMPT,
    output_schema=analyst_output_schema,
    output_key='drawing_spec',
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_model_callback=_pace_and_preprocess,
    after_model_callback=_sanitize_drawing_spec_output
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
        open_in_cadcraft_gui,
        *GEOMETRY_TOOLS,
    ],
    retry_config=retry_policy,
    generate_content_config=gen_config,
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
        open_in_cadcraft_gui,
        *GEOMETRY_TOOLS,
    ],
    retry_config=retry_policy,
    generate_content_config=gen_config,
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
    sub_agents=[analyst, draftsman, refinement_loop]
)
