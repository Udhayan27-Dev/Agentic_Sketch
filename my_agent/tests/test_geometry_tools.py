import math
import pytest
from sketchcad_agent.geometry_tools import (
    _ang,
    _unit,
    arc_angles,
    circle_circle_intersections,
    circle_line_intersections,
    fillet_two_lines,
    tangent_from_point_to_circle,
    tangent_lines_two_circles,
    line_line_intersection,
    trim_extend_lines,
    chamfer_two_lines,
    close_joint,
    trim_circle_or_arc,
    tangent_arc_line_circle,
    tangent_arc_two_circles,
    tangent_point_and_vector,
)


def test_unit_and_ang():
    u = _unit([3.0, 4.0])
    assert pytest.approx(u[0], rel=1e-5) == 0.6
    assert pytest.approx(u[1], rel=1e-5) == 0.8

    assert pytest.approx(_ang([0.0, 0.0], [10.0, 0.0])) == 0.0
    assert pytest.approx(_ang([0.0, 0.0], [0.0, 10.0])) == 90.0
    assert pytest.approx(_ang([0.0, 0.0], [-10.0, 0.0])) == 180.0
    assert pytest.approx(_ang([0.0, 0.0], [0.0, -10.0])) == 270.0


def test_fillet_two_lines_90_deg():
    corner = [0.0, 0.0]
    away1 = [10.0, 0.0]
    away2 = [0.0, 10.0]
    radius = 5.0

    res = fillet_two_lines(corner, away1, away2, radius)
    assert 'error' not in res
    assert pytest.approx(res['center'][0]) == 5.0
    assert pytest.approx(res['center'][1]) == 5.0
    assert pytest.approx(res['radius']) == 5.0
    assert pytest.approx(res['tangent_point_1'][0]) == 5.0
    assert pytest.approx(res['tangent_point_1'][1]) == 0.0
    assert pytest.approx(res['tangent_point_2'][0]) == 0.0
    assert pytest.approx(res['tangent_point_2'][1]) == 5.0


def test_fillet_two_lines_parallel():
    res = fillet_two_lines([0.0, 0.0], [1.0, 0.0], [1.0, 0.0], 5.0)
    assert 'error' in res


def test_tangent_lines_two_circles():
    c1 = [0.0, 0.0]
    r1 = 10.0
    c2 = [40.0, 0.0]
    r2 = 10.0

    outer = tangent_lines_two_circles(c1, r1, c2, r2, kind='outer')
    assert 'lines' in outer
    assert len(outer['lines']) == 2
    # For horizontal identical circles, outer tangents are horizontal y = 10 and y = -10
    ys = [line['from_circle1'][1] for line in outer['lines']]
    assert any(pytest.approx(y, abs=1e-5) == 10.0 for y in ys)
    assert any(pytest.approx(y, abs=1e-5) == -10.0 for y in ys)

    inner = tangent_lines_two_circles(c1, r1, c2, r2, kind='inner')
    assert 'lines' in inner
    assert len(inner['lines']) == 2


def test_tangent_from_point_to_circle():
    point = [20.0, 0.0]
    center = [0.0, 0.0]
    radius = 10.0

    res = tangent_from_point_to_circle(point, center, radius)
    assert 'tangent_points' in res
    pts = res['tangent_points']
    assert len(pts) == 2
    # cos(60) = 0.5 -> x = 10*0.5 = 5.0, y = +/- 10*sqrt(3)/2 = +/- 8.66025
    for p in pts:
        assert pytest.approx(p[0], abs=1e-5) == 5.0
        assert pytest.approx(abs(p[1]), abs=1e-5) == 10.0 * math.sqrt(3) / 2.0


def test_circle_circle_intersections():
    c1 = [0.0, 0.0]
    r1 = 10.0
    c2 = [10.0, 0.0]
    r2 = 10.0

    res = circle_circle_intersections(c1, r1, c2, r2)
    assert 'points' in res
    pts = res['points']
    assert len(pts) == 2
    for p in pts:
        assert pytest.approx(p[0], abs=1e-5) == 5.0
        assert pytest.approx(abs(p[1]), abs=1e-5) == math.sqrt(75.0)


def test_circle_line_intersections():
    center = [0.0, 0.0]
    radius = 10.0
    # Horizontal line at y = 6
    p1 = [-20.0, 6.0]
    p2 = [20.0, 6.0]

    res = circle_line_intersections(center, radius, p1, p2)
    assert 'points' in res
    pts = res['points']
    assert len(pts) == 2
    # x^2 + 6^2 = 10^2 -> x = +/- 8
    xs = sorted([p[0] for p in pts])
    assert pytest.approx(xs[0], abs=1e-5) == -8.0
    assert pytest.approx(xs[1], abs=1e-5) == 8.0
    assert pytest.approx(pts[0][1], abs=1e-5) == 6.0
    assert pytest.approx(pts[1][1], abs=1e-5) == 6.0

    # Non-intersecting line
    res_no = circle_line_intersections(center, radius, [-20.0, 15.0], [20.0, 15.0])
    assert res_no['points'] == []


def test_arc_angles():
    center = [0.0, 0.0]
    pa = [10.0, 0.0]    # 0 deg
    pb = [0.0, 10.0]    # 90 deg
    via = [math.cos(math.radians(45)) * 10, math.sin(math.radians(45)) * 10]  # 45 deg

    res = arc_angles(center, pa, pb, via)
    assert pytest.approx(res['start_deg']) == 0.0
    assert pytest.approx(res['end_deg']) == 90.0
    assert res['start_point'] == 'point_a'

    # Arc passing through 270 deg (the large arc)
    via_other = [0.0, -10.0]
    res_other = arc_angles(center, pa, pb, via_other)
    assert pytest.approx(res_other['start_deg']) == 90.0
    assert pytest.approx(res_other['end_deg']) == 0.0
    assert res_other['start_point'] == 'point_b'


def test_line_line_intersection():
    # Perpendicular lines meeting at (5.0, 0.0)
    res = line_line_intersection([0.0, 0.0], [10.0, 0.0], [5.0, -5.0], [5.0, 5.0])
    assert 'error' not in res
    assert pytest.approx(res['intersection'][0]) == 5.0
    assert pytest.approx(res['intersection'][1]) == 0.0
    assert res['is_on_segment1'] is True
    assert res['is_on_segment2'] is True

    # Parallel lines
    res_par = line_line_intersection([0.0, 0.0], [10.0, 0.0], [0.0, 5.0], [10.0, 5.0])
    assert res_par.get('is_parallel') is True


def test_trim_extend_lines():
    res = trim_extend_lines([0.0, 0.0], [4.0, 0.0], [5.0, 5.0], [5.0, 10.0])
    assert 'error' not in res
    assert pytest.approx(res['joint_point'][0]) == 5.0
    assert pytest.approx(res['joint_point'][1]) == 0.0
    assert res['line1_trimmed'] == [[0.0, 0.0], [5.0, 0.0]]
    assert res['line2_trimmed'] == [[5.0, 0.0], [5.0, 10.0]]


def test_chamfer_two_lines():
    corner = [0.0, 0.0]
    away1 = [10.0, 0.0]
    away2 = [0.0, 10.0]
    res = chamfer_two_lines(corner, away1, away2, d1=3.0, d2=4.0)
    assert pytest.approx(res['chamfer_point_1'][0]) == 3.0
    assert pytest.approx(res['chamfer_point_1'][1]) == 0.0
    assert pytest.approx(res['chamfer_point_2'][0]) == 0.0
    assert pytest.approx(res['chamfer_point_2'][1]) == 4.0
    assert pytest.approx(res['chamfer_length']) == 5.0


def test_close_joint():
    # Exact joint
    res_exact = close_joint([10.0, 5.0], [10.0, 5.0])
    assert res_exact['is_exact'] is True
    assert res_exact['gap_mm'] == 0.0

    # Small gap within tolerance
    res_gap = close_joint([10.0, 5.0], [10.02, 5.0], tol=0.1)
    assert res_gap['is_exact'] is False
    assert res_gap['is_within_tolerance'] is True
    assert res_gap['action'] == 'snap'
    assert pytest.approx(res_gap['unified_joint'][0]) == 10.01


def test_trim_circle_or_arc():
    center = [0.0, 0.0]
    radius = 10.0
    cp1 = [10.0, 0.0]   # 0 deg
    cp2 = [0.0, 10.0]   # 90 deg
    via = [math.cos(math.radians(45)) * 10, math.sin(math.radians(45)) * 10]

    res = trim_circle_or_arc(center, radius, cp1, cp2, via)
    assert pytest.approx(res['start_deg']) == 0.0
    assert pytest.approx(res['end_deg']) == 90.0
    assert pytest.approx(res['sweep_deg']) == 90.0
    assert pytest.approx(res['arc_length']) == 2 * math.pi * 10.0 * 0.25


def test_tangent_arc_line_circle():
    # Horizontal line along y = 0, circle at (50, 30) with r = 20
    # Blend arc with radius 10 in the corner
    line_p1 = [0.0, 0.0]
    line_p2 = [100.0, 0.0]
    c_center = [50.0, 30.0]
    c_radius = 20.0
    blend_r = 10.0

    res = tangent_arc_line_circle(line_p1, line_p2, c_center, c_radius, blend_r, near_point=[30.0, 10.0])
    assert 'error' not in res
    arc = res['best_match']
    assert pytest.approx(arc['radius']) == 10.0
    assert pytest.approx(arc['center'][1]) == 10.0
    # Distance to line y=0 is 10.0
    assert pytest.approx(arc['tangent_line_point'][1]) == 0.0
    # Distance to circle center is c_radius + blend_r = 30.0
    d_c = math.hypot(arc['center'][0] - c_center[0], arc['center'][1] - c_center[1])
    assert pytest.approx(d_c, abs=1e-4) == 30.0


def test_tangent_arc_two_circles():
    c1 = [0.0, 0.0]
    r1 = 10.0
    c2 = [40.0, 0.0]
    r2 = 10.0
    blend_r = 15.0

    res = tangent_arc_two_circles(c1, r1, c2, r2, blend_r, kind='external', near_point=[20.0, 15.0])
    assert 'error' not in res
    arc = res['best_match']
    assert pytest.approx(arc['center'][0]) == 20.0
    assert pytest.approx(arc['center'][1]) == 15.0
    # Tangent contact point 1 distance to c1 is 10.0
    d_t1 = math.hypot(arc['tangent_point_1'][0] - c1[0], arc['tangent_point_1'][1] - c1[1])
    assert pytest.approx(d_t1, abs=1e-4) == 10.0
    # Tangent contact point 2 distance to c2 is 10.0
    d_t2 = math.hypot(arc['tangent_point_2'][0] - c2[0], arc['tangent_point_2'][1] - c2[1])
    assert pytest.approx(d_t2, abs=1e-4) == 10.0


def test_tangent_point_and_vector():
    center = [0.0, 0.0]
    radius = 10.0
    # At 0 deg: point is (10, 0), CCW tangent is (0, 1) pointing +Y
    res_0 = tangent_point_and_vector(center, radius, 0.0, direction='ccw')
    assert pytest.approx(res_0['point'][0]) == 10.0
    assert pytest.approx(res_0['point'][1]) == 0.0
    assert pytest.approx(res_0['tangent_vector'][0]) == 0.0
    assert pytest.approx(res_0['tangent_vector'][1]) == 1.0

    # At 90 deg: point is (0, 10), CCW tangent is (-1, 0) pointing -X
    res_90 = tangent_point_and_vector(center, radius, 90.0, direction='ccw')
    assert pytest.approx(res_90['point'][0]) == 0.0
    assert pytest.approx(res_90['point'][1]) == 10.0
    assert pytest.approx(res_90['tangent_vector'][0]) == -1.0
    assert pytest.approx(res_90['tangent_vector'][1]) == 0.0

