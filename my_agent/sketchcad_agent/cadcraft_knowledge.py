"""CADCraft Comprehensive Knowledge Base & Agent Guidance System.

Contains full command specifications, analytical geometry equations, coordinate conventions,
and single-responsibility operational protocols for the Master Bot and Specialized Worker Agents.
"""

CADCRAFT_KNOWLEDGE_BASE = """
# CADCraft Engine Architecture & Entity Specifications

## 1. Coordinate Conventions
- Coordinates are 2D Cartesian [X, Y] in millimeters (mm).
- The primary datum (0.0, 0.0) is always anchored at the bottom-left nominal vertex of the part.
- Angles are in degrees: 0° points along +X (East), 90° points along +Y (North), 180° points along -X (West), 270° points along -Y (South). Positive angles sweep Counter-Clockwise (CCW).

## 2. Core Drawing Commands & Exact Parameter Schemas

### LINE
- Description: Draws connected line segments.
- JSON Command: "line"
- Parameters:
  {"points": [[x1, y1], [x2, y2]], "closed": false}
- Best Practice: The Envelope Agent uses lines to draw the sharp nominal profile before fillets are blended.

### CIRCLE
- Description: Places a full circular feature (boss, bore, or mounting hole).
- JSON Command: "circle"
- Parameters:
  {"center": [cx, cy], "radius": r}
- Best Practice: The Cutout Agent uses circle to cut central bores and bolt hole circles.

### ARC
- Description: Sweeps an arc from start angle to end angle (Counter-Clockwise).
- JSON Command: "arc"
- Parameters:
  {"center": [cx, cy], "radius": r, "start": start_deg, "end": end_deg}
- Best Practice: Start and end angles must be calculated via analytical arc_angles(center, p_start, p_end) to ensure mathematical tangency.

### FILLET
- Description: Rounds the sharp corner between two intersecting lines.
- JSON Command: "fillet"
- Parameters:
  {"h1": hex_handle1, "p1": [x, y], "h2": hex_handle2, "p2": [x, y], "radius": r, "trim": true}
- Alternative via Geometry Solvers: Call fillet_two_lines(p1, p_corner, p2, radius) to compute exact tangent endpoints and arc center, then draw the arc directly.

### DIMLINEAR (Linear Dimension)
- Description: Places a horizontal or vertical linear dimension callout.
- JSON Command: "dimlinear"
- Parameters:
  {"p1": [x1, y1], "p2": [x2, y2], "at": [text_x, text_y]}

### DIMRADIUS (Radius Dimension)
- Description: Annotates the radius of a circular arc or fillet.
- JSON Command: "dimradius"
- Parameters:
  {"center": [cx, cy], "point": [px, py]}

### DIMDIAMETER (Diameter Dimension)
- Description: Annotates the diameter of an internal hole or outer boss.
- JSON Command: "dimdiameter"
- Parameters:
  {"center": [cx, cy], "point": [px, py]}

## 3. The 14 Analytical Geometry Solvers (Zero Hallucination)
Under no circumstances should an agent approximate coordinates or guess trigonometry.
All continuous geometry calculations must use the analytical routines in `geometry_tools.py`:

1. fillet_two_lines(p1, p_corner, p2, radius):
   Computes exact tangent point on line 1 (t1), tangent point on line 2 (t2), and fillet center (center).
2. tangent_lines_two_circles(c1, r1, c2, r2):
   Computes exact outer and inner tangent lines between two circles.
3. tangent_from_point_to_circle(p, center, radius):
   Solves tangent lines from an external point to a circle.
4. circle_line_intersections(center, radius, p1, p2):
   Computes intersection coordinates between a circle and line for keyways and cutouts.
5. circle_circle_intersections(c1, r1, c2, r2):
   Finds exact intersection points between two overlapping arcs or circles.
6. arc_angles(center, p_start, p_end):
   Calculates start and end angles in degrees for CAD arc commands.
7. line_line_intersection(p1, p2, p3, p4):
   Computes the sharp intersection of two line segments.
8. trim_extend_lines(p1, p2, p3, p4):
   Trims or extends two lines to their exact meeting vertex.
9. chamfer_two_lines(p1, p_corner, p2, d1, d2):
   Calculates chamfer beveled transition endpoints.
10. close_joint(p_curr, p_target, tolerance=0.05):
    Snaps joint vertices together to guarantee watertightness (gap <= 0.05mm).
11. trim_circle_or_arc(center, radius, cut_angle_1, cut_angle_2, keep_ccw):
    Slices a circular feature into exact arc spans.
12. tangent_arc_line_circle(line_p1, line_p2, circle_center, circle_radius, fillet_radius):
    Calculates the blending arc between a straight line and a circle.
13. tangent_arc_two_circles(c1, r1, c2, r2, blend_radius, internal=False):
    Calculates the concave or convex blend arc bridging two separate circles.
14. tangent_point_and_vector(center, radius, angle_deg):
    Computes exact boundary point and tangent direction vector at any angle.

## 4. Multi-Agent Hierarchy & Goal Allocation

### Master Bot (Manager & Critic)
- Goal: Maintain 100% geometric accuracy.
- Roles:
  1. Semantic Perception: Analyzes input blueprint and identifies all dimensions and features.
  2. Task Decomposition: Creates isolated sub-tasks for each worker agent.
  3. Real-time Narration: Explains what is happening to the user via speech commentary.
  4. Active Pointer Director: Animates the pointer/crosshair to the coordinates being drawn.
  5. Visual & B-rep Critic: Audits each agent's work, verifies gap <= 0.05mm, and enforces tangency <= 0.1°.

### Worker Agent 1: Envelope Agent
- Single Responsibility: Draw nominal sharp outer boundary lines.
- Rules: Never attempt fillets; draw sharp corners first to preserve datum references.

### Worker Agent 2: Cutout & Bore Agent
- Single Responsibility: Place inner holes, keyway slots, and circular cutouts.
- Rules: Center coordinates must strictly reference the (0,0) datum.

### Worker Agent 3: Fillet & Blend Agent
- Single Responsibility: Apply fillets to sharp corners.
- Rules: Call fillet_two_lines to get exact tangent points and draw arc. Replace original sharp corner lines with trimmed line segments.

### Worker Agent 4: Joint & Tangency Agent
- Single Responsibility: Solve outer tangent spans and ensure closed loop watertightness.
- Rules: Guarantee every contour starts and ends at the identical coordinate with gap <= 0.05mm.

### Worker Agent 5: Dimension & Annotation Agent
- Single Responsibility: Place technical dimension lines, arrows, and numerical callouts.
- Rules: Dimension offset lines must not overlap part geometry.
"""
