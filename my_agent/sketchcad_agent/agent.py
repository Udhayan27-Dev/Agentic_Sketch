import os
import shutil
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

FREECAD_MCP_CMD = shutil.which('freecad-mcp') or str(Path(sys.executable).parent / 'freecad-mcp')

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

class DrawingSpec(BaseModel):
    units: Literal['mm']
    entities: list[Entity]
    relations: list[Relation]
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
  not compute arc endpoints or angles: the Draftsman has tools for that."""

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

freecad = McpToolset(
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

draftsman = LlmAgent(
    name='draftsman',
    model=MODEL,
    instruction=DRAFTSMAN_PROMPT,
    tools=[
        freecad,
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
    instruction=CRITIC_PROMPT,
    tools=[freecad, exit_loop],
    output_schema=CriticReport,
    output_key='critic_report',
    retry_config=retry_policy,
    generate_content_config=gen_config,
    before_model_callback=_pace_free_tier
)

refiner = LlmAgent(
    name='refiner',
    model=MODEL,
    instruction=REFINER_PROMPT,
    tools=[freecad],
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
