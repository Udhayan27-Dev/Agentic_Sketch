import math
import pytest
from sketchcad_agent.geometry_tools import (
    arc_angles,
    circle_circle_intersections,
    circle_line_intersections,
    fillet_two_lines,
    tangent_from_point_to_circle,
    tangent_lines_two_circles,
)


def test_shape_1_filleted_corner():
    """Shape 1: A filleted corner between two perpendicular lines."""
    corner = [0.0, 0.0]
    away1 = [50.0, 0.0]
    away2 = [0.0, 50.0]
    radius = 10.0

    fillet = fillet_two_lines(corner, away1, away2, radius)
    assert 'error' not in fillet
    assert pytest.approx(fillet['center']) == [10.0, 10.0]
    assert pytest.approx(fillet['radius']) == 10.0
    assert pytest.approx(fillet['tangent_point_1']) == [10.0, 0.0]
    assert pytest.approx(fillet['tangent_point_2']) == [0.0, 10.0]

    # Arc runs counter-clockwise between 180 and 270 degrees
    assert pytest.approx(fillet['start_deg']) == 180.0
    assert pytest.approx(fillet['end_deg']) == 270.0


def test_shape_2_slot():
    """Shape 2: A slot consisting of two flat lines and two end semicircles."""
    c1 = [10.0, 5.0]
    c2 = [40.0, 5.0]
    radius = 5.0

    # Top line y = 10 from x = 10 to 40
    # Bottom line y = 0 from x = 40 to 10
    # Semicircle 1 at left: from [10, 0] to [10, 10] via [5, 5]
    left_arc = arc_angles(c1, [10.0, 10.0], [10.0, 0.0], via_point=[5.0, 5.0])
    # CCW from [10, 10] (90 deg) through [5, 5] (180 deg) to [10, 0] (270 deg)
    assert pytest.approx(left_arc['start_deg']) == 90.0
    assert pytest.approx(left_arc['end_deg']) == 270.0

    # Semicircle 2 at right: from [40, 10] to [40, 0] via [45, 5]
    right_arc = arc_angles(c2, [40.0, 10.0], [40.0, 0.0], via_point=[45.0, 5.0])
    # CCW from [40, 0] (270 deg) through [45, 5] (0 deg) to [40, 10] (90 deg)
    assert pytest.approx(right_arc['start_deg']) == 270.0
    assert pytest.approx(right_arc['end_deg']) == 90.0


def test_shape_3_line_tangent_to_circle():
    """Shape 3: A line from point (50, 0) tangent to circle center (0, 0) R=20."""
    pt = [50.0, 0.0]
    center = [0.0, 0.0]
    radius = 20.0

    res = tangent_from_point_to_circle(pt, center, radius)
    assert 'tangent_points' in res
    t_pts = res['tangent_points']
    assert len(t_pts) == 2
    # Distance from center to tangent point must equal radius
    for tp in t_pts:
        dist = math.hypot(tp[0] - center[0], tp[1] - center[1])
        assert pytest.approx(dist, abs=1e-5) == radius
        # Line from pt to tp must be perpendicular to radius from center to tp
        rad_vec = [tp[0] - center[0], tp[1] - center[1]]
        tan_vec = [pt[0] - tp[0], pt[1] - tp[1]]
        dot = rad_vec[0] * tan_vec[0] + rad_vec[1] * tan_vec[1]
        assert pytest.approx(dot, abs=1e-5) == 0.0


def test_shape_4_circle_with_notch():
    """Shape 4: Main circle (0,0) R=30 cut by notch circle (30, 0) R=10."""
    c1 = [0.0, 0.0]
    r1 = 30.0
    c2 = [30.0, 0.0]
    r2 = 10.0

    inter = circle_circle_intersections(c1, r1, c2, r2)
    assert 'points' in inter
    pts = inter['points']
    assert len(pts) == 2

    # Main kept arc passes through (-30, 0)
    main_arc = arc_angles(c1, pts[0], pts[1], via_point=[-30.0, 0.0])
    assert 'start_deg' in main_arc and 'end_deg' in main_arc

    # Notch arc kept passes through (20, 0)
    notch_arc = arc_angles(c2, pts[0], pts[1], via_point=[20.0, 0.0])
    assert 'start_deg' in notch_arc and 'end_deg' in notch_arc


def test_shape_5_link_two_circles_outer_tangents():
    """Shape 5: Two circles joined by outer tangents (link/pulley shape)."""
    c1 = [0.0, 0.0]
    r1 = 15.0
    c2 = [60.0, 0.0]
    r2 = 25.0

    tans = tangent_lines_two_circles(c1, r1, c2, r2, kind='outer')
    assert 'lines' in tans
    lines = tans['lines']
    assert len(lines) == 2

    p1_top = lines[0]['from_circle1']
    p2_top = lines[0]['to_circle2']
    p1_bot = lines[1]['from_circle1']
    p2_bot = lines[1]['to_circle2']

    # Circle 1 kept arc passes through [-15, 0]
    arc1 = arc_angles(c1, p1_top, p1_bot, via_point=[-15.0, 0.0])
    # Circle 2 kept arc passes through [85, 0]
    arc2 = arc_angles(c2, p2_top, p2_bot, via_point=[85.0, 0.0])

    assert 'start_deg' in arc1 and 'end_deg' in arc1
    assert 'start_deg' in arc2 and 'end_deg' in arc2


def test_flat_lines_meeting_circle():
    """Tests flat line meeting circle profile (e.g. seal/profile outlines)."""
    center = [50.0, 45.0]
    radius = 55.0
    # Flat horizontal line y = 10
    line_p1 = [0.0, 10.0]
    line_p2 = [100.0, 10.0]

    inter = circle_line_intersections(center, radius, line_p1, line_p2)
    assert len(inter['points']) == 2
    pt1, pt2 = inter['points']

    # Keep top arc via (50, 100)
    arc = arc_angles(center, pt1, pt2, via_point=[50.0, 100.0])
    assert 'start_deg' in arc and 'end_deg' in arc
