import math


def _unit(v: list[float]) -> list[float]:
    n = math.hypot(v[0], v[1])
    return [v[0] / n, v[1] / n]


def _ang(c: list[float], p: list[float]) -> float:
    return math.degrees(math.atan2(p[1] - c[1], p[0] - c[0])) % 360


def fillet_two_lines(
    corner: list[float], away1: list[float], away2: list[float], radius: float
) -> dict:
    """Rounded corner between two lines meeting at corner. away1 and away2
    are direction vectors pointing from the corner along each line. Draw
    each line only up to its returned tangent point."""
    u1, u2 = _unit(away1), _unit(away2)
    full = math.acos(max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1])))
    half = full / 2
    if not (1e-3 < half < math.pi / 2 - 1e-3):
        return {'error': 'lines are parallel or the corner is too flat'}
    t = radius / math.tan(half)
    bis = _unit([u1[0] + u2[0], u1[1] + u2[1]])
    d = radius / math.sin(half)
    c = [corner[0] + bis[0] * d, corner[1] + bis[1] * d]
    t1 = [corner[0] + u1[0] * t, corner[1] + u1[1] * t]
    t2 = [corner[0] + u2[0] * t, corner[1] + u2[1] * t]
    a, b, swapped = _ang(c, t1), _ang(c, t2), False
    if (b - a) % 360 > 180:
        a, b, swapped = b, a, True
    return {
        'center': c,
        'radius': radius,
        'tangent_point_1': t1,
        'tangent_point_2': t2,
        'start_deg': a,
        'end_deg': b,
        'start_joins': 'line2' if swapped else 'line1',
        'end_joins': 'line1' if swapped else 'line2',
        'note': 'arc runs counter-clockwise from start_deg to end_deg',
    }


def tangent_lines_two_circles(
    c1: list[float], r1: float, c2: list[float], r2: float, kind: str = 'outer'
) -> dict:
    """Two tangent lines between two circles. kind: outer or inner."""
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dy)
    k = (r1 - r2) if kind == 'outer' else (r1 + r2)
    if d <= abs(k) + 1e-9:
        return {'error': 'no such tangent: circles overlap or contain each other'}
    alpha, beta = math.atan2(dy, dx), math.acos(k / d)
    sign = 1 if kind == 'outer' else -1
    lines = []
    for s in (1, -1):
        n = [math.cos(alpha + s * beta), math.sin(alpha + s * beta)]
        lines.append({
            'from_circle1': [c1[0] + r1 * n[0], c1[1] + r1 * n[1]],
            'to_circle2': [c2[0] + sign * r2 * n[0], c2[1] + sign * r2 * n[1]],
        })
    return {'lines': lines}


def tangent_from_point_to_circle(
    point: list[float], center: list[float], radius: float
) -> dict:
    """The two points where a line from point touches the circle tangentially."""
    dx, dy = point[0] - center[0], point[1] - center[1]
    d = math.hypot(dx, dy)
    if d <= radius + 1e-9:
        return {'error': 'point is inside or on the circle'}
    a, b = math.atan2(dy, dx), math.acos(radius / d)
    return {
        'tangent_points': [
            [
                center[0] + radius * math.cos(a + s * b),
                center[1] + radius * math.sin(a + s * b),
            ]
            for s in (1, -1)
        ]
    }


def circle_circle_intersections(
    c1: list[float], r1: float, c2: list[float], r2: float
) -> dict:
    """Intersection points of two circles (use instead of trim)."""
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d = math.hypot(dx, dy)
    if d < 1e-9 or d > r1 + r2 or d < abs(r1 - r2):
        return {'points': []}
    a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
    h = math.sqrt(max(r1 * r1 - a * a, 0.0))
    mx, my = c1[0] + a * dx / d, c1[1] + a * dy / d
    return {
        'points': [
            [mx + h * dy / d, my - h * dx / d],
            [mx - h * dy / d, my + h * dx / d],
        ]
    }


def circle_line_intersections(
    center: list[float], radius: float, p1: list[float], p2: list[float]
) -> dict:
    """Points where the line through p1 and p2 meets the circle."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - center[0], p1[1] - center[1]
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4 * a * c
    if a < 1e-12 or disc < 0:
        return {'points': []}
    s = math.sqrt(disc)
    return {
        'points': [
            [p1[0] + t * dx, p1[1] + t * dy]
            for t in ((-b - s) / (2 * a), (-b + s) / (2 * a))
        ]
    }


def arc_angles(
    center: list[float],
    point_a: list[float],
    point_b: list[float],
    via_point: list[float],
) -> dict:
    """Start and end angle of the counter-clockwise arc between point_a and
    point_b that passes through via_point. Use to pick which side to keep."""
    a, b, v = _ang(center, point_a), _ang(center, point_b), _ang(center, via_point)
    if ((v - a) % 360) <= ((b - a) % 360):
        return {'start_deg': a, 'end_deg': b, 'start_point': 'point_a'}
    return {'start_deg': b, 'end_deg': a, 'start_point': 'point_b'}
