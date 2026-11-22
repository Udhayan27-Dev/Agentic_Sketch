"""Calibrated Exemplar Patterns for CADCraft Arcs, Tangents, Fillets, and Trims.

Used for in-context grounding and few-shot conditioning of the Analyst and Draftsman agents.
"""

CADCRAFT_FEW_SHOT_EXEMPLARS = """
================================================================================
EXEMPLAR 0: PART DECOMPOSITION & (0.0, 0.0) ORIGIN ANCHORING
================================================================================
Rule: Every drawing MUST establish an explicit origin datum (0.0, 0.0) first.
Every single vertex, coordinate, circle center, dimension, and annotation MUST be calculated
and referenced strictly relative to (0.0, 0.0).

Functional Part Breakdown Procedure:
1. Part 1 [base_envelope]: Nominal sharp boundary edges before corner fillets.
2. Part 2 [circular_feature]: Circular hubs, arches, central bores with centers [cx, cy] relative to (0,0).
3. Part 3 [cutout_joint]: Internal slots, notches, keyway cuts that intersect boundaries or circles.
4. Part 4 [fillet_round]: Corner roundings applied to sharp edges.
5. Part 5 [annotation]: Stated dimensions and text labels.

================================================================================
EXEMPLAR 1: HOW TO APPLY FILLETS (Draw Sharp Edges First -> Fillet Tool -> Curve Output)
================================================================================
Scenario: Plate with base width 100 mm, height 60 mm, with R=15.0 mm fillet at bottom-right corner.
Procedure:
Step 1: Identify nominal sharp corner vertex C = [100.0, 0.0] where nominal bottom line meets right vertical line.
Step 2: Call fillet_two_lines(corner=[100.0, 0.0], away1=[-1.0, 0.0], away2=[0.0, 1.0], radius=15.0).
   Returns:
   - tangent_point_1: [85.0, 0.0]  (Where bottom line stops)
   - tangent_point_2: [100.0, 15.0] (Where right line begins)
   - center: [85.0, 15.0], radius: 15.0
   - start_deg: 270.0, end_deg: 360.0 (or 0.0)
Step 3: Execute CADCraft commands:
   execute(command="line", params={"points": [[0.0, 0.0], [85.0, 0.0]]})
   execute(command="arc", params={"center": [85.0, 15.0], "radius": 15.0, "start": 270.0, "end": 360.0})
   execute(command="line", params={"points": [[100.0, 15.0], [100.0, 60.0]]})
   execute(command="dimradius", params={"center": [85.0, 15.0], "point": [100.0, 15.0]})
Result: Sharp edge is converted to an exact, watertight curve blend with zero gap.

================================================================================
EXEMPLAR 2: CIRCLE CUTS AND JOINTS STEP-BY-STEP (Hub Bore with Keyway Slot)
================================================================================
Scenario: Hub center C = [50.0, 50.0], Bore radius R = 20.0 mm.
Rectangular keyway notch width W = 6.0 mm, depth D = 4.0 mm cutting into the top of the bore.
Procedure:
Step 1: Determine keyway boundary relative to origin (0,0):
   - Left vertical edge at x = 50.0 - 3.0 = 47.0
   - Right vertical edge at x = 50.0 + 3.0 = 53.0
   - Top horizontal edge at y = 50.0 + 20.0 + 4.0 = 74.0
Step 2: Calculate exact circle intersection points using circle_line_intersections:
   - Call circle_line_intersections(center=[50.0, 50.0], radius=20.0, p1=[47.0, 0.0], p2=[47.0, 100.0])
     Yields left penetration cut point: p_cut_l = [47.0, 69.7744]
   - Call circle_line_intersections(center=[50.0, 50.0], radius=20.0, p1=[53.0, 0.0], p2=[53.0, 100.0])
     Yields right penetration cut point: p_cut_r = [53.0, 69.7744]
Step 3: Calculate kept bore arc sweep around bottom using arc_angles:
   - Call arc_angles(center=[50.0, 50.0], p_start=p_cut_r, p_end=p_cut_l, via_point=[50.0, 30.0])
     Yields start_deg: 81.3731, end_deg: 98.6269 (sweeping counter-clockwise across bottom 342.75 deg)
Step 4: Execute CADCraft commands to build the watertight closed joint:
   execute(command="line", params={"points": [p_cut_l, [47.0, 74.0], [53.0, 74.0], p_cut_r]})
   execute(command="arc", params={"center": [50.0, 50.0], "radius": 20.0, "start": 81.3731, "end": 98.6269})
   execute(command="dimlinear", params={"p1": [47.0, 74.0], "p2": [53.0, 74.0], "at": [50.0, 82.0]})
   execute(command="dimlinear", params={"p1": [50.0, 70.0], "p2": [50.0, 74.0], "at": [62.0, 72.0]})
Step 5: Verify joint continuity:
   - Call close_joint(p_end=p_cut_r, p_start=p_cut_r) -> Confirmed watertight (gap = 0.000 mm).

================================================================================
EXEMPLAR 3: TANGENT BELT (Two Circles Connected by Tangent Lines)
================================================================================
Scenario: Left hub center C1=[35.0, 35.0] R1=35.0 mm, Right hub center C2=[175.0, 35.0] R2=18.0 mm.
Procedure:
1. Call tangent_lines_two_circles(c1=[35.0, 35.0], r1=35.0, c2=[175.0, 35.0], r2=18.0, kind="outer").
   Returns top tangent points [p_c1_top, p_c2_top] and bottom tangent points [p_c1_bot, p_c2_bot].
2. Call arc_angles for each circle to get the exact counter-clockwise kept arc sweep:
   - Left hub arc: from p_c1_bot to p_c1_top via [0.0, 35.0].
   - Right hub arc: from p_c2_top to p_c2_bot via [193.0, 35.0].
3. Execute CADCraft commands:
   execute(command="line", params={"points": [p_c1_top, p_c2_top]})
   execute(command="arc", params={"center": [175.0, 35.0], "radius": 18.0, "start": ang2_start, "end": ang2_end})
   execute(command="line", params={"points": [p_c2_bot, p_c1_bot]})
   execute(command="arc", params={"center": [35.0, 35.0], "radius": 35.0, "start": ang1_start, "end": ang1_end})
   execute(command="circle", params={"center": [35.0, 35.0], "radius": 15.0})
   execute(command="circle", params={"center": [175.0, 35.0], "radius": 6.0})
   execute(command="dimlinear", params={"p1": [35.0, 35.0], "p2": [175.0, 35.0], "at": [105.0, 85.0]})

================================================================================
EXEMPLAR 4: SLOTTED OBLONG LINK WITH CONCENTRIC CUTOUTS
================================================================================
Scenario: Oblong link anchored at origin (0,0).
Left hub center C1=[25.0, 25.0] R=25.0 mm, Right hub center C2=[105.0, 25.0] R=25.0 mm.
Concentric inner slot with radius r_slot = 10.0 mm between [25.0, 25.0] and [105.0, 25.0].
Procedure:
1. Outer contour:
   execute(command="line", params={"points": [[25.0, 0.0], [105.0, 0.0]]})
   execute(command="arc", params={"center": [105.0, 25.0], "radius": 25.0, "start": 270.0, "end": 90.0})
   execute(command="line", params={"points": [[105.0, 50.0], [25.0, 50.0]]})
   execute(command="arc", params={"center": [25.0, 25.0], "radius": 25.0, "start": 90.0, "end": 270.0})
2. Inner slot cutout:
   execute(command="line", params={"points": [[25.0, 15.0], [105.0, 15.0]]})
   execute(command="arc", params={"center": [105.0, 25.0], "radius": 10.0, "start": 270.0, "end": 90.0})
   execute(command="line", params={"points": [[105.0, 35.0], [25.0, 35.0]]})
   execute(command="arc", params={"center": [25.0, 25.0], "radius": 10.0, "start": 90.0, "end": 270.0})
3. Dimensions:
   execute(command="dimlinear", params={"p1": [25.0, 25.0], "p2": [105.0, 25.0], "at": [65.0, -15.0]})
   execute(command="dimradius", params={"center": [105.0, 25.0], "point": [130.0, 25.0]})

================================================================================
EXEMPLAR 5: MECHANICAL FLANGE (Complete Parts Separation -> Sharp Edges -> Fillets -> Cuts)
================================================================================
Scenario: Stepped Flange Bracket with base 100x20 mm, top neck 40x30 mm, R=5 mm corner fillets, and Ø20 mm central hole.
Part Decomposition:
- Origin (0,0): Bottom-left datum vertex.
- Part 1 (Base Envelope): Sharp nominal envelope vertices [[0,0], [100,0], [100,20], [70,20], [70,50], [30,50], [30,20], [0,20]].
- Part 2 (Fillet Rounds):
  * Fillet F1 at [100,0] (away [-1,0], [0,1], R=5) -> trim to [95,0] and [100,5]
  * Fillet F2 at [70,20] (internal re-entrant fillet, away [1,0], [0,1], R=5)
  * Fillet F3 at [30,20] (internal re-entrant fillet, away [0,1], [-1,0], R=5)
  * Fillet F4 at [0,0] (away [0,1], [1,0], R=5) -> trim to [0,5] and [5,0]
- Part 3 (Internal Bore Cutout): Circle at [50, 35], radius = 10.0 mm.
- Part 4 (Dimensions): Linear dimensions for base width (100 mm), total height (50 mm), and hole diameter (Ø20 mm).
Procedure:
1. Apply fillet_two_lines to each corner vertex.
2. Build outer closed contour using resulting line segments and fillet arcs.
3. Draw central bore circle.
4. Add linear and radial dimensions.

================================================================================
EXEMPLAR 6: CRISP CORNERS & OVERKILL (FILLET R 0 & DEDUPLICATION)
================================================================================
Rule: To connect two non-touching, gap-separated, or overshooting lines into an exact 90° or angled sharp corner:
1. Use FILLET R 0 (or fillet_two_lines with radius=0.0):
   - Trims back overshooting lines or extends undershooting lines to their exact intersection point.
   - Eliminates manual trigonometry or guesswork when calculating corner intersections.
2. Deduplication with OVERKILL:
   - If repeated or overlapping line segments occur, CADCraft's OVERKILL routine merges collinear segments and deletes duplicates to guarantee a clean entity database.

================================================================================
EXEMPLAR 7: CIRCLE CUTTING VIA TRIM & BREAK (COUNTER-CLOCKWISE RULE)
================================================================================
Rule: In CADCraft, circular entities are parameterized from 0° to 360° counter-clockwise (CCW).
1. D-Shaft Flat Cut:
   - Hub center C=[50.0, 50.0], shaft radius R=20.0 mm, flat cut at distance d=15.0 mm from center.
   - Cutting chord line from [36.77, 65.0] to [63.23, 65.0].
   - Kept arc: CCW from [63.23, 65.0] around the bottom through [50.0, 30.0] to [36.77, 65.0].
   - Watertight profile: execute line across the chord, execute arc across the kept circle portion.
2. Semicircular Bite / Scallop on Plate Edge:
   - Plate top edge y=60.0 mm. Semicircular notch radius R=12.0 mm centered at [50.0, 60.0].
   - Straight edge stops at left contact [38.0, 60.0] and resumes at right contact [62.0, 60.0].
   - Inward notch arc: center [50.0, 60.0], radius 12.0 mm, sweeping CCW from 180° to 360° (or 0°).
3. BREAK Command CCW Convention:
   - To cut a gap directly out of a circle without cutting lines, BREAK cuts CCW from First Pick to Second Pick.
   - Example: To cut a 90° top-right gap out of a circle, pick 0° first, then pick 90° second.
4. Internal Holes in Plates:
   - Enclosed circle entities inside an outer closed boundary automatically act as negative bores/holes in CADCraft topology.
   - Never trim plate boundaries for internal holes; place circle directly at [cx, cy] with radius R.

================================================================================
EXEMPLAR 8: U-CLEVIS / FORK WITH ROUNDED LUGS & INNER SEMICIRCLE POCKET
================================================================================
Scenario: Symmetric U-clevis, outer width 60 mm, base height 20 mm, prongs extending to total height 70 mm.
Prong width 15 mm each (left prong x=0 to 15, right prong x=45 to 60).
Internal slot width 30 mm (x=15 to 45), slot depth 50 mm, with internal semicircular bottom R=15 mm.
Prong tips rounded with full semicircular arcs R=7.5 mm.
Procedure:
1. Origin (0.0, 0.0) at bottom-left corner of the clevis base.
2. Left outer contour:
   execute(command="line", params={"points": [[0.0, 0.0], [0.0, 62.5]]})
   execute(command="arc", params={"center": [7.5, 62.5], "radius": 7.5, "start": 180.0, "end": 0.0})
   execute(command="line", params={"points": [[15.0, 62.5], [15.0, 35.0]]})
3. Internal semicircular slot pocket:
   execute(command="arc", params={"center": [30.0, 35.0], "radius": 15.0, "start": 180.0, "end": 360.0})
4. Right inner contour & lug tip:
   execute(command="line", params={"points": [[45.0, 35.0], [45.0, 62.5]]})
   execute(command="arc", params={"center": [52.5, 62.5], "radius": 7.5, "start": 180.0, "end": 0.0})
   execute(command="line", params={"points": [[60.0, 62.5], [60.0, 0.0]]})
   execute(command="line", params={"points": [[60.0, 0.0], [0.0, 0.0]]})

================================================================================
EXEMPLAR 9: CIRCULAR FLANGE WITH PCD BOLT RING (ARRAYPOLAR)
================================================================================
Scenario: Circular flange outer Ø120 mm, central bore Ø40 mm with 6 mm keyway, and 6 bolt holes Ø10 mm on Ø80 mm Pitch Circle Diameter (PCD).
Procedure:
1. Datum origin at flange center [0.0, 0.0] or bottom-left corner [60.0, 60.0].
   Using center datum C=[0.0, 0.0]:
2. Outer rim: execute(command="circle", params={"center": [0.0, 0.0], "radius": 60.0})
3. Central bore with keyway (R=20.0 mm, keyway W=6 mm, depth 3 mm):
   Cut penetration points at x=±3.0 mm; kept bore arc + keyway lines connected watertight.
4. Bolt Hole Array on PCD Ø80 (radius 40.0 mm):
   - Bolt 1 at [40.0, 0.0]: execute(command="circle", params={"center": [40.0, 0.0], "radius": 5.0})
   - Polar array: execute(command="arraypolar", params={"center": [0.0, 0.0], "items": 6, "angle": 360.0})
   - Or direct coordinate generation: 6 holes at [40*cos(i*60°), 40*sin(i*60°)] for i=0..5.

================================================================================
EXEMPLAR 10: DISTORTED / SKEWED HAND-DRAWN IMAGE REGULARIZATION
================================================================================
Scenario: Input image is a smartphone photo of a hand-drawn sketch with optical skew, perspective tilt, and wavy strokes.
AI Agent Regularization Protocol:
1. Never trace raw pixel coordinates or pixel aspect ratios from distorted photos.
2. Read nominal engineering dimensions from text/OCR (e.g. "100", "60", "R12", "Ø25").
3. Establish canonical origin at primary datum corner: (0.0, 0.0).
4. Strictly snap nominal horizontal edges to y = const and vertical edges to x = const (0.0° and 90.0°).
5. Align all concentric features to their true mathematical centers.
6. Enforce exact tangency at line-arc transitions using mathematical solvers (fillet_two_lines, tangent_lines_two_circles).
7. Watertightness rule: Verify every closed loop has end-to-end tolerance <= 0.05 mm using close_joint.

================================================================================
EXEMPLAR 11: MCP PROTOCOL CONTRACT & SAFE TIMEOUT RECOVERY
================================================================================
Operational Rules:
1. Session Init: Always call new_drawing(metric=True) for millimeter drawings.
2. Tool Dispatch:
   - Use execute for deterministic single commands with structured JSON parameters.
   - Use command_line for AutoCAD-style interactive sequences.
   - Use script for multi-line batch scripts (.scr).
3. Safe Timeout Recovery:
   - If timeout returns state: "not-run", retrying the command is safe.
   - If timeout returns state: "may-have-run", NEVER blindly retry—it may create duplicate entities on the UI thread!
   - Always call inspect_drawing or query_entities first to verify whether the entity handle was created.
4. Dual Verification:
   - Geometric: Confirm bounding extents match nominal dimensions in inspect_drawing.
   - Visual: Call render(width=1024, height=768, fit=True, plot_style="monochrome.ctb") to audit visual fidelity.
"""
