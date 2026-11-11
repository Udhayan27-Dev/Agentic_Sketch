import math


def _unit(v: list[float]) -> list[float]:
    n = math.hypot(v[0], v[1])
    if n < 1e-12:
        return [0.0, 0.0]
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


def line_line_intersection(
    p1: list[float], p2: list[float], p3: list[float], p4: list[float]
) -> dict:
    """Exact 2D intersection point between Line 1 (passing through p1, p2)
    and Line 2 (passing through p3, p4). Use for exact corner joints and trimming."""
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    det = d1x * d2y - d1y * d2x
    if abs(det) < 1e-11:
        return {'error': 'lines are parallel or collinear', 'is_parallel': True}
    dx, dy = p3[0] - p1[0], p3[1] - p1[1]
    t1 = (dx * d2y - dy * d2x) / det
    t2 = (dx * d1y - dy * d1x) / det
    ix = p1[0] + t1 * d1x
    iy = p1[1] + t1 * d1y
    return {
        'intersection': [round(ix, 6), round(iy, 6)],
        't1': round(t1, 6),
        't2': round(t2, 6),
        'is_parallel': False,
        'is_on_segment1': -1e-6 <= t1 <= 1.0 + 1e-6,
        'is_on_segment2': -1e-6 <= t2 <= 1.0 + 1e-6,
    }


def trim_extend_lines(
    l1_start: list[float], l1_end: list[float], l2_start: list[float], l2_end: list[float]
) -> dict:
    """Trims or extends two meeting line segments to their exact corner joint intersection point.
    Returns the unified corner vertex and the trimmed line segments."""
    inter = line_line_intersection(l1_start, l1_end, l2_start, l2_end)
    if 'error' in inter:
        return inter
    pt = inter['intersection']
    return {
        'joint_point': pt,
        'line1_trimmed': [l1_start, pt],
        'line2_trimmed': [pt, l2_end],
        't1': inter['t1'],
        't2': inter['t2'],
        'note': 'Draw line 1 from l1_start to joint_point, and line 2 from joint_point to l2_end'
    }


def chamfer_two_lines(
    corner: list[float], away1: list[float], away2: list[float], d1: float, d2: float = None
) -> dict:
    """Beveled corner joint (chamfer) between two lines meeting at corner.
    away1 and away2 point along each line away from the corner.
    d1 and d2 are setback distances from the corner along line 1 and line 2."""
    if d2 is None:
        d2 = d1
    u1, u2 = _unit(away1), _unit(away2)
    cp1 = [corner[0] + u1[0] * d1, corner[1] + u1[1] * d1]
    cp2 = [corner[0] + u2[0] * d2, corner[1] + u2[1] * d2]
    c_len = math.hypot(cp2[0] - cp1[0], cp2[1] - cp1[1])
    return {
        'corner': corner,
        'chamfer_point_1': [round(cp1[0], 6), round(cp1[1], 6)],
        'chamfer_point_2': [round(cp2[0], 6), round(cp2[1], 6)],
        'chamfer_line': [[round(cp1[0], 6), round(cp1[1], 6)], [round(cp2[0], 6), round(cp2[1], 6)]],
        'chamfer_length': round(c_len, 6),
        'note': 'Draw line 1 up to chamfer_point_1, connect chamfer line to chamfer_point_2, then draw line 2'
    }


def close_joint(p_end: list[float], p_start: list[float], tol: float = 0.5) -> dict:
    """Validates and snaps two adjacent entity endpoints to ensure an exact closed joint.
    Prevents open contours and gaps in CAD drawings."""
    gap = math.hypot(p_start[0] - p_end[0], p_start[1] - p_end[1])
    mid = [(p_end[0] + p_start[0]) / 2.0, (p_end[1] + p_start[1]) / 2.0]
    return {
        'is_exact': gap < 1e-6,
        'is_within_tolerance': gap <= tol,
        'gap_mm': round(gap, 6),
        'unified_joint': [round(mid[0], 6), round(mid[1], 6)],
        'action': 'none' if gap < 1e-6 else ('snap' if gap <= tol else 'large_gap_detected')
    }


def trim_circle_or_arc(
    center: list[float],
    radius: float,
    cut_point_1: list[float],
    cut_point_2: list[float],
    keep_via_point: list[float],
) -> dict:
    """Trims a circle or arc between two cutting boundary points (cut_point_1 and cut_point_2),
    keeping the portion that passes through keep_via_point.
    Returns exact counter-clockwise arc parameters (start_deg, end_deg, start_point, end_point)."""
    res = arc_angles(center, cut_point_1, cut_point_2, keep_via_point)
    start_deg = res['start_deg']
    end_deg = res['end_deg']
    start_pt = cut_point_1 if res['start_point'] == 'point_a' else cut_point_2
    end_pt = cut_point_2 if res['start_point'] == 'point_a' else cut_point_1
    sweep = (end_deg - start_deg) % 360
    if sweep == 0.0:
        sweep = 360.0
    arc_len = (sweep / 360.0) * (2.0 * math.pi * radius)
    return {
        'center': center,
        'radius': radius,
        'start_point': [round(start_pt[0], 6), round(start_pt[1], 6)],
        'end_point': [round(end_pt[0], 6), round(end_pt[1], 6)],
        'start_deg': round(start_deg, 4),
        'end_deg': round(end_deg, 4),
        'sweep_deg': round(sweep, 4),
        'arc_length': round(arc_len, 6),
        'note': 'Arc runs counter-clockwise from start_deg to end_deg'
    }


def tangent_arc_line_circle(
    line_p1: list[float],
    line_p2: list[float],
    circle_center: list[float],
    circle_radius: float,
    blend_radius: float,
    near_point: list[float] = None,
) -> dict:
    """Calculates a fillet/blend arc of radius blend_radius that is tangent to BOTH
    a straight line (passing through line_p1 and line_p2) and a circle (circle_center, circle_radius).
    Ideal for rounded transitions between straight edges and circular heads.
    If near_point is provided, selects the closest candidate blend arc."""
    dx, dy = line_p2[0] - line_p1[0], line_p2[1] - line_p1[1]
    L = math.hypot(dx, dy)
    if L < 1e-12:
        return {'error': 'line length is zero'}
    u = [dx / L, dy / L]
    n = [-u[1], u[0]]

    candidates = []
    expanded_r = circle_radius + blend_radius
    for sign in (1.0, -1.0):
        off_p1 = [line_p1[0] + sign * blend_radius * n[0], line_p1[1] + sign * blend_radius * n[1]]
        off_p2 = [off_p1[0] + u[0], off_p1[1] + u[1]]
        inter = circle_line_intersections(circle_center, expanded_r, off_p1, off_p2)
        for center in inter['points']:
            proj_t = (center[0] - line_p1[0]) * u[0] + (center[1] - line_p1[1]) * u[1]
            t_line = [line_p1[0] + proj_t * u[0], line_p1[1] + proj_t * u[1]]
            c_dx, c_dy = center[0] - circle_center[0], center[1] - circle_center[1]
            c_dist = math.hypot(c_dx, c_dy)
            if c_dist < 1e-12:
                continue
            t_circle = [circle_center[0] + (circle_radius / c_dist) * c_dx, circle_center[1] + (circle_radius / c_dist) * c_dy]
            a1 = _ang(center, t_line)
            a2 = _ang(center, t_circle)
            candidates.append({
                'center': [round(center[0], 6), round(center[1], 6)],
                'radius': blend_radius,
                'tangent_line_point': [round(t_line[0], 6), round(t_line[1], 6)],
                'tangent_circle_point': [round(t_circle[0], 6), round(t_circle[1], 6)],
                'angle_line_deg': round(a1, 4),
                'angle_circle_deg': round(a2, 4),
            })

    if not candidates:
        return {'error': 'no valid tangent blend arc exists'}
    if near_point:
        candidates.sort(key=lambda c: math.hypot(c['center'][0] - near_point[0], c['center'][1] - near_point[1]))
    return {'blend_arcs': candidates, 'best_match': candidates[0]}


def tangent_arc_two_circles(
    c1: list[float],
    r1: float,
    c2: list[float],
    r2: float,
    blend_radius: float,
    kind: str = 'external',
    near_point: list[float] = None,
) -> dict:
    """Calculates a blend arc of radius blend_radius tangent to two circles/arcs.
    Used for external fillets, internal blends, or reverse curves (S-curves/bulges).
    kind options:
      - 'external': fillet between two separate outer contours (R1 = r1 + R, R2 = r2 + R)
      - 'internal': blend enclosing both circles
      - 'reverse_1': external to c1, internal to c2
      - 'reverse_2': internal to c1, external to c2
    If near_point is provided, selects the closest candidate blend arc."""
    if kind == 'external':
        R1, R2 = r1 + blend_radius, r2 + blend_radius
        sign1, sign2 = 1.0, 1.0
    elif kind == 'internal':
        R1, R2 = abs(r1 - blend_radius), abs(r2 - blend_radius)
        sign1, sign2 = -1.0, -1.0
    elif kind == 'reverse_1':
        R1, R2 = r1 + blend_radius, abs(r2 - blend_radius)
        sign1, sign2 = 1.0, -1.0
    elif kind == 'reverse_2':
        R1, R2 = abs(r1 - blend_radius), r2 + blend_radius
        sign1, sign2 = -1.0, 1.0
    else:
        return {'error': f'unknown kind {kind}'}

    inter = circle_circle_intersections(c1, R1, c2, R2)
    if not inter['points']:
        return {'error': 'no tangent arc exists for given radii and centers'}

    candidates = []
    for center in inter['points']:
        d1 = math.hypot(center[0] - c1[0], center[1] - c1[1])
        t1 = [c1[0] + sign1 * (r1 / d1) * (center[0] - c1[0]), c1[1] + sign1 * (r1 / d1) * (center[1] - c1[1])]
        d2 = math.hypot(center[0] - c2[0], center[1] - c2[1])
        t2 = [c2[0] + sign2 * (r2 / d2) * (center[0] - c2[0]), c2[1] + sign2 * (r2 / d2) * (center[1] - c2[1])]

        candidates.append({
            'center': [round(center[0], 6), round(center[1], 6)],
            'radius': blend_radius,
            'tangent_point_1': [round(t1[0], 6), round(t1[1], 6)],
            'tangent_point_2': [round(t2[0], 6), round(t2[1], 6)],
            'angle_1_deg': round(_ang(center, t1), 4),
            'angle_2_deg': round(_ang(center, t2), 4),
        })

    if near_point:
        candidates.sort(key=lambda c: math.hypot(c['center'][0] - near_point[0], c['center'][1] - near_point[1]))
    return {'blend_arcs': candidates, 'best_match': candidates[0]}


def tangent_point_and_vector(
    center: list[float], radius: float, angle_deg: float, direction: str = 'ccw'
) -> dict:
    """Returns the exact contact point on an arc/circle at angle_deg, along with the
    unit tangent direction vector (ccw or cw) and normal unit vector.
    Ideal for starting a tangent line smoothly off an arc endpoint."""
    rad = math.radians(angle_deg)
    px = center[0] + radius * math.cos(rad)
    py = center[1] + radius * math.sin(rad)
    nx, ny = math.cos(rad), math.sin(rad)
    if direction == 'ccw':
        tx, ty = -math.sin(rad), math.cos(rad)
    else:
        tx, ty = math.sin(rad), -math.cos(rad)
    return {
        'point': [round(px, 6), round(py, 6)],
        'tangent_vector': [round(tx, 6), round(ty, 6)],
        'normal_vector': [round(nx, 6), round(ny, 6)],
        'angle_deg': angle_deg,
    }
