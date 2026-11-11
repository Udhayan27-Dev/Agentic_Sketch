"""Calibrated Exemplar Patterns for CADCraft Arcs, Tangents, Fillets, and Trims.

Used for in-context grounding and few-shot conditioning of the Analyst and Draftsman agents.
"""

CADCRAFT_FEW_SHOT_EXEMPLARS = """
================================================================================
EXEMPLAR 1: CORNER FILLETS (Sharp Lines -> Smooth Arc Transition)
================================================================================
Scenario: Two lines meeting at a corner (e.g. (100, 0)) with fillet radius R = 15.0 mm.
Procedure:
1. Call fillet_two_lines(corner=[100.0, 0.0], away1=[-1.0, 0.0], away2=[0.0, 1.0], radius=15.0).
   Returns:
   - tangent_point_1: [85.0, 0.0]
   - tangent_point_2: [100.0, 15.0]
   - center: [85.0, 15.0]
   - start_deg: 270.0, end_deg: 360.0 (or 0.0)
2. Execute CADCraft commands:
   execute(command="line", params={"points": [[0.0, 0.0], [85.0, 0.0]]})
   execute(command="arc", params={"center": [85.0, 15.0], "radius": 15.0, "start": 270.0, "end": 360.0})
   execute(command="line", params={"points": [[100.0, 15.0], [100.0, 60.0]]})
   execute(command="dimradius", params={"center": [85.0, 15.0], "point": [100.0, 15.0]})

================================================================================
EXEMPLAR 2: TANGENT BELT (Two Circles Connected by Tangent Lines)
================================================================================
Scenario: Left hub center C1=[35, 35] R1=35 mm, Right hub center C2=[175, 35] R2=18 mm.
Procedure:
1. Call tangent_lines_two_circles(c1=[35, 35], r1=35, c2=[175, 35], r2=18, kind="outer").
   Returns top tangent [p_c1_top, p_c2_top] and bottom tangent [p_c1_bot, p_c2_bot].
2. Call arc_angles for each circle to get the exact counter-clockwise kept arc sweep.
3. Execute CADCraft commands:
   execute(command="line", params={"points": [p_c1_top, p_c2_top]})
   execute(command="arc", params={"center": [175, 35], "radius": 18.0, "start": ang2_start, "end": ang2_end})
   execute(command="line", params={"points": [p_c2_bot, p_c1_bot]})
   execute(command="arc", params={"center": [35, 35], "radius": 35.0, "start": ang1_start, "end": ang1_end})
   execute(command="circle", params={"center": [35, 35], "radius": 15.0})
   execute(command="circle", params={"center": [175, 35], "radius": 6.0})
   execute(command="dimlinear", params={"p1": [35, 35], "p2": [175, 35], "at": [105, 80]})

================================================================================
EXEMPLAR 3: TRIMS & NOTCHES (Bore with Keyway Cutout)
================================================================================
Scenario: Bore of radius R=20.0 mm with rectangular keyway notch width W=6.0 mm, depth D=4.0 mm.
Procedure:
1. Calculate intersections of keyway vertical edges with circle using circle_line_intersections.
   Yields cut points p_cut_l and p_cut_r on circle perimeter.
2. Call arc_angles with center, p_cut_r, p_cut_l, and via_point=[cx, cy - r] (the bottom of the circle).
   Yields exact start_deg and end_deg of the kept circular arc.
3. Execute CADCraft commands:
   execute(command="line", params={"points": [p_cut_l, p_corner_l, p_corner_r, p_cut_r]})
   execute(command="arc", params={"center": center, "radius": 20.0, "start": start_deg, "end": end_deg})
   execute(command="dimlinear", params={"p1": p_corner_l, "p2": p_corner_r, "at": [cx, y_top + 10]})

================================================================================
EXEMPLAR 4: SLOTTED LINK (Parallel Lines Blending into Semicircles)
================================================================================
Scenario: Oblong link between centers C1=[25, 25] and C2=[105, 25], R=25 mm.
Procedure:
1. Execute CADCraft commands:
   execute(command="line", params={"points": [[25.0, 0.0], [105.0, 0.0]]})
   execute(command="arc", params={"center": [105.0, 25.0], "radius": 25.0, "start": 270.0, "end": 90.0})
   execute(command="line", params={"points": [[105.0, 50.0], [25.0, 50.0]]})
   execute(command="arc", params={"center": [25.0, 25.0], "radius": 25.0, "start": 90.0, "end": 270.0})
   execute(command="dimlinear", params={"p1": [25.0, 25.0], "p2": [105.0, 25.0], "at": [65.0, -20.0]})
"""
