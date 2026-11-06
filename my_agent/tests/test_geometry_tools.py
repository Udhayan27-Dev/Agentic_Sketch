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
