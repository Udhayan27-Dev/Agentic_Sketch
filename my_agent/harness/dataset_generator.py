"""Parametric Dataset Generator for CAD Technical Drawings with Arcs, Tangents, Fillets, and Trims.

Generates paired datasets consisting of:
1. High-resolution synthetic engineering drawing (.png) with dimensions and annotations.
2. Ground-truth DrawingSpec (Pydantic schema).
3. Executable CADCraft commands and scripts (.json / .scr).
4. Verified ground-truth DXF and rendered CADCraft PNG via cadcraft-cli.
"""

import json
import math
import os
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from sketchcad_agent.geometry_tools import (
    arc_angles,
    circle_circle_intersections,
    circle_line_intersections,
    fillet_two_lines,
    tangent_lines_two_circles,
    tangent_from_point_to_circle,
)


@dataclass
class CADEntity:
    id: str
    type: str  # 'line', 'circle', 'arc'
    p1: Optional[List[float]] = None
    p2: Optional[List[float]] = None
    center: Optional[List[float]] = None
    radius: Optional[float] = None
    start_deg: Optional[float] = None
    end_deg: Optional[float] = None


@dataclass
class CADRelation:
    kind: str  # 'tangent', 'fillet', 'coincident', 'distance', 'radius', 'horizontal', 'vertical'
    refs: List[str]
    value_mm: Optional[float] = None


@dataclass
class CADAnnotation:
    text: str
    position: List[float]
    height: float = 3.0
    rotation_deg: float = 0.0


@dataclass
class DimensionCallout:
    kind: str  # 'linear_h', 'linear_v', 'aligned', 'radius', 'diameter'
    p1: List[float]
    p2: List[float]
    label: str
    offset: float = 12.0
    center: Optional[List[float]] = None
    radius: Optional[float] = None


@dataclass
class GeneratedSample:
    sample_id: str
    category: str
    description: str
    entities: List[CADEntity]
    relations: List[CADRelation]
    annotations: List[CADAnnotation]
    dimensions: List[DimensionCallout]
    cadcraft_commands: List[Dict[str, Any]]
    cadcraft_script: str
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


def draw_arrowhead(draw: ImageDraw.ImageDraw, tip: Tuple[float, float], direction: Tuple[float, float],
                   arrow_len: float = 10.0, arrow_width: float = 3.5, fill: str = "black"):
    """Draw a sharp engineering filled arrowhead pointing towards tip."""
    dx, dy = direction
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux  # perpendicular
    base_x = tip[0] - ux * arrow_len
    base_y = tip[1] - uy * arrow_len
    p1 = tip
    p2 = (base_x + nx * arrow_width, base_y + ny * arrow_width)
    p3 = (base_x - nx * arrow_width, base_y - ny * arrow_width)
    draw.polygon([p1, p2, p3], fill=fill)


def render_drawing_image(sample: GeneratedSample, output_path: str, img_width: int = 1200, img_height: int = 900) -> str:
    """Renders a production-grade 2D CAD technical drawing on white canvas with dimensions and annotations."""
    min_x, min_y, max_x, max_y = sample.bounds
    span_x = max(max_x - min_x, 10.0)
    span_y = max(max_y - min_y, 10.0)

    margin = 120
    draw_w = img_width - 2 * margin
    draw_h = img_height - 2 * margin

    scale = min(draw_w / (span_x * 1.5), draw_h / (span_y * 1.5))
    # Center the drawing
    mid_x = (min_x + max_x) / 2.0
    mid_y = (min_y + max_y) / 2.0
    canvas_cx = img_width / 2.0
    canvas_cy = img_height / 2.0

    def cad_to_px(x: float, y: float) -> Tuple[float, float]:
        px = canvas_cx + (x - mid_x) * scale
        py = canvas_cy - (y - mid_y) * scale  # Y-up in CAD to Y-down in image
        return (px, py)

    img = Image.new("RGB", (img_width, img_height), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.load_default(size=20)
        font_dim = ImageFont.load_default(size=15)
        font_small = ImageFont.load_default(size=13)
    except TypeError:
        font_large = ImageFont.load_default()
        font_dim = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title Block / Header
    draw.text((margin, 35), f"TECHNICAL DRAWING: {sample.category.upper()}", fill="#111111", font=font_large)
    draw.text((margin, 65), f"Sample ID: {sample.sample_id} | Units: mm | Origin anchored strictly at (0.0, 0.0)", fill="#555555", font=font_small)
    draw.line([(margin, 90), (img_width - margin, 90)], fill="#888888", width=1)

    # Origin marker (0, 0)
    o_px = cad_to_px(0.0, 0.0)
    draw.ellipse([(o_px[0] - 5, o_px[1] - 5), (o_px[0] + 5, o_px[1] + 5)], outline="#D32F2F", width=2)
    draw.line([(o_px[0] - 10, o_px[1]), (o_px[0] + 10, o_px[1])], fill="#D32F2F", width=1)
    draw.line([(o_px[0], o_px[1] - 10), (o_px[0], o_px[1] + 10)], fill="#D32F2F", width=1)
    draw.text((o_px[0] + 8, o_px[1] + 4), "(0,0)", fill="#D32F2F", font=font_small)

    # 1. Draw Centerlines for circles and arcs
    for ent in sample.entities:
        if ent.type in ('circle', 'arc') and ent.center is not None and ent.radius is not None:
            c_px = cad_to_px(ent.center[0], ent.center[1])
            r_px = ent.radius * scale
            # Crosshair center lines
            ext = max(r_px + 12, 20)
            draw.line([(c_px[0] - ext, c_px[1]), (c_px[0] + ext, c_px[1])], fill="#999999", width=1)
            draw.line([(c_px[0], c_px[1] - ext), (c_px[0], c_px[1] + ext)], fill="#999999", width=1)

    # 2. Draw Object Lines (Contours)
    for ent in sample.entities:
        if ent.type == 'line' and ent.p1 and ent.p2:
            p1_px = cad_to_px(ent.p1[0], ent.p1[1])
            p2_px = cad_to_px(ent.p2[0], ent.p2[1])
            draw.line([p1_px, p2_px], fill="#111111", width=3)
        elif ent.type == 'circle' and ent.center and ent.radius:
            cx, cy = ent.center
            r = ent.radius
            p_min = cad_to_px(cx - r, cy + r)
            p_max = cad_to_px(cx + r, cy - r)
            draw.ellipse([p_min, p_max], outline="#111111", width=3)
        elif ent.type == 'arc' and ent.center and ent.radius and ent.start_deg is not None and ent.end_deg is not None:
            cx, cy = ent.center
            r = ent.radius
            p_min = cad_to_px(cx - r, cy + r)
            p_max = cad_to_px(cx + r, cy - r)
            # In CAD, angle 0 is +X, 90 is +Y (counter-clockwise).
            # In PIL, angle 0 is +X, 90 is +Y (clockwise, down).
            # So CAD angle phi corresponds to -phi in PIL, or arc from -end_deg to -start_deg.
            a_start = ent.start_deg % 360
            a_end = ent.end_deg % 360
            # Normalize counter-clockwise sweep
            span = (a_end - a_start) % 360
            if span == 0:
                span = 360
            pil_start = (-a_start - span) % 360
            pil_end = (-a_start) % 360
            draw.arc([p_min, p_max], start=pil_start, end=pil_end, fill="#111111", width=3)

    # 3. Draw Dimension Lines, Extension Lines, and Dimension Texts
    for dim in sample.dimensions:
        if dim.kind == 'linear_h':
            # Horizontal dimension between p1 and p2
            x1, y1 = dim.p1
            x2, y2 = dim.p2
            dim_y = max(y1, y2) + dim.offset if dim.offset >= 0 else min(y1, y2) + dim.offset
            p1_obj = cad_to_px(x1, y1)
            p2_obj = cad_to_px(x2, y2)
            p1_dim = cad_to_px(x1, dim_y)
            p2_dim = cad_to_px(x2, dim_y)
            # Extension lines
            draw.line([p1_obj, p1_dim], fill="#777777", width=1)
            draw.line([p2_obj, p2_dim], fill="#777777", width=1)
            # Dimension line with arrows
            draw.line([p1_dim, p2_dim], fill="#333333", width=2)
            draw_arrowhead(draw, p1_dim, (p1_dim[0] - p2_dim[0], 0), arrow_len=8, arrow_width=3, fill="#333333")
            draw_arrowhead(draw, p2_dim, (p2_dim[0] - p1_dim[0], 0), arrow_len=8, arrow_width=3, fill="#333333")
            # Text label
            mid_px = ((p1_dim[0] + p2_dim[0]) / 2, p1_dim[1] - 18)
            draw.text(mid_px, dim.label, fill="#000000", font=font_dim)

        elif dim.kind == 'linear_v':
            # Vertical dimension between p1 and p2
            x1, y1 = dim.p1
            x2, y2 = dim.p2
            dim_x = max(x1, x2) + dim.offset if dim.offset >= 0 else min(x1, x2) + dim.offset
            p1_obj = cad_to_px(x1, y1)
            p2_obj = cad_to_px(x2, y2)
            p1_dim = cad_to_px(dim_x, y1)
            p2_dim = cad_to_px(dim_x, y2)
            # Extension lines
            draw.line([p1_obj, p1_dim], fill="#777777", width=1)
            draw.line([p2_obj, p2_dim], fill="#777777", width=1)
            # Dimension line with arrows
            draw.line([p1_dim, p2_dim], fill="#333333", width=2)
            draw_arrowhead(draw, p1_dim, (0, p1_dim[1] - p2_dim[1]), arrow_len=8, arrow_width=3, fill="#333333")
            draw_arrowhead(draw, p2_dim, (0, p2_dim[1] - p1_dim[1]), arrow_len=8, arrow_width=3, fill="#333333")
            # Text label
            mid_py = (p1_dim[0] + 6, (p1_dim[1] + p2_dim[1]) / 2 - 8)
            draw.text(mid_py, dim.label, fill="#000000", font=font_dim)

        elif dim.kind in ('radius', 'diameter'):
            # Radial dimension leader line
            cx, cy = dim.center if dim.center else (0.0, 0.0)
            r = dim.radius if dim.radius else 10.0
            ang_rad = math.radians(45.0)
            px = cx + r * math.cos(ang_rad)
            py = cy + r * math.sin(ang_rad)
            p_on = cad_to_px(px, py)
            p_leader = (p_on[0] + 35, p_on[1] - 25)
            p_tail = (p_leader[0] + 25, p_leader[1])

            draw.line([p_on, p_leader, p_tail], fill="#333333", width=2)
            draw_arrowhead(draw, p_on, (p_on[0] - p_leader[0], p_on[1] - p_leader[1]), arrow_len=8, arrow_width=3, fill="#333333")
            draw.text((p_tail[0] + 5, p_tail[1] - 8), dim.label, fill="#000000", font=font_dim)

    # 4. Text Annotations
    for ann in sample.annotations:
        pos_px = cad_to_px(ann.position[0], ann.position[1])
        draw.text(pos_px, ann.text, fill="#222222", font=font_small)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    return output_path


class ShapeGenerator:
    """Parametric shape generator for 5 distinct mechanical shape families."""

    @staticmethod
    def generate_corner_fillet_plate(sample_id: str, seed: Optional[int] = None) -> GeneratedSample:
        """Family 1: Rectangle or Stepped Plate with Exact Corner Fillets."""
        rng = random.Random(seed)
        width = rng.choice([80.0, 90.0, 100.0, 120.0, 140.0])
        height = rng.choice([50.0, 60.0, 70.0, 80.0])
        r_fillet = rng.choice([8.0, 10.0, 12.0, 15.0])
        hole_r = rng.choice([6.0, 8.0, 10.0])

        # Bottom-left anchored at (0, 0)
        # Sharp corners:
        # C0=(0,0), C1=(width,0), C2=(width,height), C3=(0,height)
        # Fillet 4 corners:
        # F0 (at C0): corner=(0,0), away1=(1,0), away2=(0,1)
        f0 = fillet_two_lines([0.0, 0.0], [1.0, 0.0], [0.0, 1.0], r_fillet)
        # F1 (at C1): corner=(width,0), away1=(0,1), away2=(-1,0)
        f1 = fillet_two_lines([width, 0.0], [0.0, 1.0], [-1.0, 0.0], r_fillet)
        # F2 (at C2): corner=(width,height), away1=(-1,0), away2=(0,-1)
        f2 = fillet_two_lines([width, height], [-1.0, 0.0], [0.0, -1.0], r_fillet)
        # F3 (at C3): corner=(0,height), away1=(0,-1), away2=(1,0)
        f3 = fillet_two_lines([0.0, height], [0.0, -1.0], [1.0, 0.0], r_fillet)

        entities = [
            # Base bottom line
            CADEntity(id="e1", type="line", p1=f0['tangent_point_1'], p2=f1['tangent_point_2']),
            # Fillet 1 (bottom right)
            CADEntity(id="e2", type="arc", center=f1['center'], radius=r_fillet, start_deg=f1['start_deg'], end_deg=f1['end_deg']),
            # Right vertical line
            CADEntity(id="e3", type="line", p1=f1['tangent_point_1'], p2=f2['tangent_point_2']),
            # Fillet 2 (top right)
            CADEntity(id="e4", type="arc", center=f2['center'], radius=r_fillet, start_deg=f2['start_deg'], end_deg=f2['end_deg']),
            # Top horizontal line
            CADEntity(id="e5", type="line", p1=f2['tangent_point_1'], p2=f3['tangent_point_2']),
            # Fillet 3 (top left)
            CADEntity(id="e6", type="arc", center=f3['center'], radius=r_fillet, start_deg=f3['start_deg'], end_deg=f3['end_deg']),
            # Left vertical line
            CADEntity(id="e7", type="line", p1=f3['tangent_point_1'], p2=f0['tangent_point_2']),
            # Fillet 0 (bottom left)
            CADEntity(id="e8", type="arc", center=f0['center'], radius=r_fillet, start_deg=f0['start_deg'], end_deg=f0['end_deg']),
            # Concentric center mounting hole
            CADEntity(id="e9", type="circle", center=[width / 2.0, height / 2.0], radius=hole_r),
        ]

        relations = [
            CADRelation(kind="horizontal", refs=["e1"]),
            CADRelation(kind="tangent", refs=["e1", "e2"]),
            CADRelation(kind="vertical", refs=["e3"]),
            CADRelation(kind="tangent", refs=["e3", "e4"]),
            CADRelation(kind="horizontal", refs=["e5"]),
            CADRelation(kind="tangent", refs=["e5", "e6"]),
            CADRelation(kind="vertical", refs=["e7"]),
            CADRelation(kind="tangent", refs=["e7", "e8"]),
            CADRelation(kind="tangent", refs=["e8", "e1"]),
            CADRelation(kind="fillet", refs=["e1", "e3"], value_mm=r_fillet),
            CADRelation(kind="fillet", refs=["e3", "e5"], value_mm=r_fillet),
            CADRelation(kind="fillet", refs=["e5", "e7"], value_mm=r_fillet),
            CADRelation(kind="fillet", refs=["e7", "e1"], value_mm=r_fillet),
        ]

        dimensions = [
            DimensionCallout(kind="linear_h", p1=[0.0, 0.0], p2=[width, 0.0], label=f"{width:.1f} mm", offset=-18.0),
            DimensionCallout(kind="linear_v", p1=[width, 0.0], p2=[width, height], label=f"{height:.1f} mm", offset=20.0),
            DimensionCallout(kind="radius", p1=[], p2=[], label=f"4x R{r_fillet:.1f}", center=f2['center'], radius=r_fillet),
            DimensionCallout(kind="diameter", p1=[], p2=[], label=f"Ø{hole_r*2:.1f}", center=[width/2, height/2], radius=hole_r),
        ]

        annotations = [
            CADAnnotation(text=f"CORNER FILLET PLATE R{r_fillet:.1f}", position=[10.0, height + 15.0]),
            CADAnnotation(text="ALL FILLETS R TYP 4 PLCS", position=[width / 2.0 - 25.0, -28.0]),
        ]

        # CADCraft command execution sequence
        cadcraft_cmds = [
            {"cmd": "line", "params": {"points": [f0['tangent_point_1'], f1['tangent_point_2']]}},
            {"cmd": "arc.ser", "params": {"start": f1['tangent_point_2'], "end": f1['tangent_point_1'], "radius": r_fillet}},
            {"cmd": "line", "params": {"points": [f1['tangent_point_1'], f2['tangent_point_2']]}},
            {"cmd": "arc.ser", "params": {"start": f2['tangent_point_2'], "end": f2['tangent_point_1'], "radius": r_fillet}},
            {"cmd": "line", "params": {"points": [f2['tangent_point_1'], f3['tangent_point_2']]}},
            {"cmd": "arc.ser", "params": {"start": f3['tangent_point_2'], "end": f3['tangent_point_1'], "radius": r_fillet}},
            {"cmd": "line", "params": {"points": [f3['tangent_point_1'], f0['tangent_point_2']]}},
            {"cmd": "arc.ser", "params": {"start": f0['tangent_point_2'], "end": f0['tangent_point_1'], "radius": r_fillet}},
            {"cmd": "circle", "params": {"center": [width / 2.0, height / 2.0], "radius": hole_r}},
            {"cmd": "dimlinear", "params": {"p1": [0.0, 0.0], "p2": [width, 0.0], "at": [width / 2.0, -18.0]}},
            {"cmd": "dimlinear", "params": {"p1": [width, 0.0], "p2": [width, height], "at": [width + 20.0, height / 2.0]}},
            {"cmd": "dimradius", "params": {"center": f2['center'], "point": [f2['center'][0] + r_fillet, f2['center'][1]]}},
        ]

        scr_lines = [
            f"LINE {f0['tangent_point_1'][0]},{f0['tangent_point_1'][1]} {f1['tangent_point_2'][0]},{f1['tangent_point_2'][1]} ",
            f"ARC C {f1['center'][0]},{f1['center'][1]} {f1['tangent_point_2'][0]},{f1['tangent_point_2'][1]} {f1['tangent_point_1'][0]},{f1['tangent_point_1'][1]}",
            f"LINE {f1['tangent_point_1'][0]},{f1['tangent_point_1'][1]} {f2['tangent_point_2'][0]},{f2['tangent_point_2'][1]} ",
            f"ARC C {f2['center'][0]},{f2['center'][1]} {f2['tangent_point_2'][0]},{f2['tangent_point_2'][1]} {f2['tangent_point_1'][0]},{f2['tangent_point_1'][1]}",
            f"LINE {f2['tangent_point_1'][0]},{f2['tangent_point_1'][1]} {f3['tangent_point_2'][0]},{f3['tangent_point_2'][1]} ",
            f"ARC C {f3['center'][0]},{f3['center'][1]} {f3['tangent_point_2'][0]},{f3['tangent_point_2'][1]} {f3['tangent_point_1'][0]},{f3['tangent_point_1'][1]}",
            f"LINE {f3['tangent_point_1'][0]},{f3['tangent_point_1'][1]} {f0['tangent_point_2'][0]},{f0['tangent_point_2'][1]} ",
            f"ARC C {f0['center'][0]},{f0['center'][1]} {f0['tangent_point_2'][0]},{f0['tangent_point_2'][1]} {f0['tangent_point_1'][0]},{f0['tangent_point_1'][1]}",
            f"CIRCLE {width/2.0},{height/2.0} {hole_r}",
        ]

        return GeneratedSample(
            sample_id=sample_id,
            category="corner_fillets",
            description=f"Corner Filleted Plate {width:.0f}x{height:.0f} with R{r_fillet:.0f} fillets and Ø{hole_r*2:.0f} bore",
            entities=entities,
            relations=relations,
            annotations=annotations,
            dimensions=dimensions,
            cadcraft_commands=cadcraft_cmds,
            cadcraft_script="\n".join(scr_lines) + "\n",
            bounds=(0.0, 0.0, width, height),
        )

    @staticmethod
    def generate_tangent_pulley(sample_id: str, seed: Optional[int] = None) -> GeneratedSample:
        """Family 2: Two Circular Hubs connected by Outer Tangent Lines (Pulley / Rocker Arm)."""
        rng = random.Random(seed)
        r1 = rng.choice([25.0, 30.0, 35.0, 40.0])
        r2 = rng.choice([15.0, 18.0, 20.0, 24.0])
        dist = rng.choice([80.0, 100.0, 120.0, 140.0])
        bore1 = rng.choice([10.0, 12.0, 15.0])
        bore2 = rng.choice([6.0, 8.0, 10.0])

        c1 = [r1, r1]
        c2 = [r1 + dist, r1]

        tan_res = tangent_lines_two_circles(c1, r1, c2, r2, kind="outer")
        lines = tan_res['lines']
        # Line 0: top tangent line (from circle 1 to circle 2)
        # Line 1: bottom tangent line
        top_tan = lines[0]
        bot_tan = lines[1]

        p_c1_top = top_tan['from_circle1']
        p_c2_top = top_tan['to_circle2']
        p_c1_bot = bot_tan['from_circle1']
        p_c2_bot = bot_tan['to_circle2']

        # Hub 1 Arc: runs counter-clockwise from bottom tangent point around left to top tangent point
        ang_c1_res = arc_angles(c1, p_c1_bot, p_c1_top, [c1[0] - r1, c1[1]])
        # Hub 2 Arc: runs counter-clockwise from top tangent point around right to bottom tangent point
        ang_c2_res = arc_angles(c2, p_c2_top, p_c2_bot, [c2[0] + r2, c2[1]])

        entities = [
            # Top tangent line
            CADEntity(id="e1", type="line", p1=p_c1_top, p2=p_c2_top),
            # Right Hub Arc (Counter-Clockwise)
            CADEntity(id="e2", type="arc", center=c2, radius=r2, start_deg=ang_c2_res['start_deg'], end_deg=ang_c2_res['end_deg']),
            # Bottom tangent line (from circle 2 back to circle 1)
            CADEntity(id="e3", type="line", p1=p_c2_bot, p2=p_c1_bot),
            # Left Hub Arc (Counter-Clockwise)
            CADEntity(id="e4", type="arc", center=c1, radius=r1, start_deg=ang_c1_res['start_deg'], end_deg=ang_c1_res['end_deg']),
            # Left bore hole
            CADEntity(id="e5", type="circle", center=c1, radius=bore1),
            # Right bore hole
            CADEntity(id="e6", type="circle", center=c2, radius=bore2),
        ]

        relations = [
            CADRelation(kind="tangent", refs=["e4", "e1"]),
            CADRelation(kind="tangent", refs=["e1", "e2"]),
            CADRelation(kind="tangent", refs=["e2", "e3"]),
            CADRelation(kind="tangent", refs=["e3", "e4"]),
            CADRelation(kind="distance", refs=["e5", "e6"], value_mm=dist),
        ]

        dimensions = [
            DimensionCallout(kind="linear_h", p1=c1, p2=c2, label=f"{dist:.1f} mm CTRS", offset=r1 + 25.0),
            DimensionCallout(kind="radius", p1=[], p2=[], label=f"R{r1:.1f}", center=c1, radius=r1),
            DimensionCallout(kind="radius", p1=[], p2=[], label=f"R{r2:.1f}", center=c2, radius=r2),
            DimensionCallout(kind="diameter", p1=[], p2=[], label=f"Ø{bore1*2:.1f}", center=c1, radius=bore1),
            DimensionCallout(kind="diameter", p1=[], p2=[], label=f"Ø{bore2*2:.1f}", center=c2, radius=bore2),
        ]

        annotations = [
            CADAnnotation(text="TANGENT BELT PROFILE (TYP 2 PLCS)", position=[c1[0], r1 * 2 + 15.0]),
        ]

        cadcraft_cmds = [
            {"cmd": "line", "params": {"points": [p_c1_top, p_c2_top]}},
            {"cmd": "arc", "params": {"center": c2, "radius": r2, "start": ang_c2_res['start_deg'], "end": ang_c2_res['end_deg']}},
            {"cmd": "line", "params": {"points": [p_c2_bot, p_c1_bot]}},
            {"cmd": "arc", "params": {"center": c1, "radius": r1, "start": ang_c1_res['start_deg'], "end": ang_c1_res['end_deg']}},
            {"cmd": "circle", "params": {"center": c1, "radius": bore1}},
            {"cmd": "circle", "params": {"center": c2, "radius": bore2}},
            {"cmd": "dimlinear", "params": {"p1": c1, "p2": c2, "at": [(c1[0] + c2[0]) / 2, c1[1] + r1 + 25.0]}},
            {"cmd": "dimradius", "params": {"center": c1, "point": [c1[0] - r1, c1[1]]}},
            {"cmd": "dimradius", "params": {"center": c2, "point": [c2[0] + r2, c2[1]]}},
        ]

        scr_lines = [
            f"LINE {p_c1_top[0]},{p_c1_top[1]} {p_c2_top[0]},{p_c2_top[1]} ",
            f"ARC C {c2[0]},{c2[1]} {p_c2_top[0]},{p_c2_top[1]} {p_c2_bot[0]},{p_c2_bot[1]}",
            f"LINE {p_c2_bot[0]},{p_c2_bot[1]} {p_c1_bot[0]},{p_c1_bot[1]} ",
            f"ARC C {c1[0]},{c1[1]} {p_c1_bot[0]},{p_c1_bot[1]} {p_c1_top[0]},{p_c1_top[1]}",
            f"CIRCLE {c1[0]},{c1[1]} {bore1}",
            f"CIRCLE {c2[0]},{c2[1]} {bore2}",
        ]

        return GeneratedSample(
            sample_id=sample_id,
            category="tangent_profile",
            description=f"Tangent Pulley/Rocker Arm L={dist:.0f} R1={r1:.0f} R2={r2:.0f}",
            entities=entities,
            relations=relations,
            annotations=annotations,
            dimensions=dimensions,
            cadcraft_commands=cadcraft_cmds,
            cadcraft_script="\n".join(scr_lines) + "\n",
            bounds=(0.0, 0.0, c2[0] + r2, r1 * 2),
        )

    @staticmethod
    def generate_slotted_link(sample_id: str, seed: Optional[int] = None) -> GeneratedSample:
        """Family 3: Oblong Slotted Link with Semicircular End Arcs and Internal Slot."""
        rng = random.Random(seed)
        link_len = rng.choice([70.0, 80.0, 100.0, 120.0])
        r_outer = rng.choice([20.0, 25.0, 30.0])
        slot_len = rng.choice([30.0, 40.0, 50.0])
        r_slot = rng.choice([6.0, 8.0, 10.0])

        c_left = [r_outer, r_outer]
        c_right = [r_outer + link_len, r_outer]

        # Outer contour:
        # Bottom line from c_left bottom to c_right bottom
        p_bot_l = [c_left[0], 0.0]
        p_bot_r = [c_right[0], 0.0]
        # Right 180 deg arc from bottom to top
        p_top_r = [c_right[0], r_outer * 2.0]
        # Top line from c_right top to c_left top
        p_top_l = [c_left[0], r_outer * 2.0]
        # Left 180 deg arc from top to bottom

        # Internal slot:
        mid_x = (c_left[0] + c_right[0]) / 2.0
        s_left_c = [mid_x - slot_len / 2.0, r_outer]
        s_right_c = [mid_x + slot_len / 2.0, r_outer]

        entities = [
            CADEntity(id="e1", type="line", p1=p_bot_l, p2=p_bot_r),
            CADEntity(id="e2", type="arc", center=c_right, radius=r_outer, start_deg=270.0, end_deg=90.0),
            CADEntity(id="e3", type="line", p1=p_top_r, p2=p_top_l),
            CADEntity(id="e4", type="arc", center=c_left, radius=r_outer, start_deg=90.0, end_deg=270.0),
            # Internal slot entities
            CADEntity(id="e5", type="line", p1=[s_left_c[0], r_outer - r_slot], p2=[s_right_c[0], r_outer - r_slot]),
            CADEntity(id="e6", type="arc", center=s_right_c, radius=r_slot, start_deg=270.0, end_deg=90.0),
            CADEntity(id="e7", type="line", p1=[s_right_c[0], r_outer + r_slot], p2=[s_left_c[0], r_outer + r_slot]),
            CADEntity(id="e8", type="arc", center=s_left_c, radius=r_slot, start_deg=90.0, end_deg=270.0),
        ]

        relations = [
            CADRelation(kind="tangent", refs=["e1", "e2"]),
            CADRelation(kind="tangent", refs=["e2", "e3"]),
            CADRelation(kind="tangent", refs=["e3", "e4"]),
            CADRelation(kind="tangent", refs=["e4", "e1"]),
            CADRelation(kind="tangent", refs=["e5", "e6"]),
            CADRelation(kind="tangent", refs=["e6", "e7"]),
            CADRelation(kind="tangent", refs=["e7", "e8"]),
            CADRelation(kind="tangent", refs=["e8", "e5"]),
        ]

        dimensions = [
            DimensionCallout(kind="linear_h", p1=c_left, p2=c_right, label=f"{link_len:.1f} mm CTRS", offset=-22.0),
            DimensionCallout(kind="linear_h", p1=s_left_c, p2=s_right_c, label=f"SLOT {slot_len:.1f}", offset=15.0),
            DimensionCallout(kind="radius", p1=[], p2=[], label=f"2x R{r_outer:.1f}", center=c_right, radius=r_outer),
            DimensionCallout(kind="radius", p1=[], p2=[], label=f"2x R{r_slot:.1f}", center=s_left_c, radius=r_slot),
        ]

        annotations = [
            CADAnnotation(text="OBLONG SLOTTED LINK", position=[10.0, r_outer * 2 + 15.0]),
        ]

        cadcraft_cmds = [
            {"cmd": "line", "params": {"points": [p_bot_l, p_bot_r]}},
            {"cmd": "arc", "params": {"center": c_right, "radius": r_outer, "start": 270.0, "end": 90.0}},
            {"cmd": "line", "params": {"points": [p_top_r, p_top_l]}},
            {"cmd": "arc", "params": {"center": c_left, "radius": r_outer, "start": 90.0, "end": 270.0}},
            # Slot
            {"cmd": "line", "params": {"points": [[s_left_c[0], r_outer - r_slot], [s_right_c[0], r_outer - r_slot]]}},
            {"cmd": "arc", "params": {"center": s_right_c, "radius": r_slot, "start": 270.0, "end": 90.0}},
            {"cmd": "line", "params": {"points": [[s_right_c[0], r_outer + r_slot], [s_left_c[0], r_outer + r_slot]]}},
            {"cmd": "arc", "params": {"center": s_left_c, "radius": r_slot, "start": 90.0, "end": 270.0}},
            {"cmd": "dimlinear", "params": {"p1": c_left, "p2": c_right, "at": [mid_x, -22.0]}},
            {"cmd": "dimradius", "params": {"center": c_right, "point": [c_right[0] + r_outer, c_right[1]]}},
        ]

        scr_lines = [
            f"LINE {p_bot_l[0]},{p_bot_l[1]} {p_bot_r[0]},{p_bot_r[1]} ",
            f"ARC C {c_right[0]},{c_right[1]} {p_bot_r[0]},{p_bot_r[1]} {p_top_r[0]},{p_top_r[1]}",
            f"LINE {p_top_r[0]},{p_top_r[1]} {p_top_l[0]},{p_top_l[1]} ",
            f"ARC C {c_left[0]},{c_left[1]} {p_top_l[0]},{p_top_l[1]} {p_bot_l[0]},{p_bot_l[1]}",
            f"LINE {s_left_c[0]},{r_outer - r_slot} {s_right_c[0]},{r_outer - r_slot} ",
            f"ARC C {s_right_c[0]},{s_right_c[1]} {s_right_c[0]},{r_outer - r_slot} {s_right_c[0]},{r_outer + r_slot}",
            f"LINE {s_right_c[0]},{r_outer + r_slot} {s_left_c[0]},{r_outer + r_slot} ",
            f"ARC C {s_left_c[0]},{s_left_c[1]} {s_left_c[0]},{r_outer + r_slot} {s_left_c[0]},{r_outer - r_slot}",
        ]

        return GeneratedSample(
            sample_id=sample_id,
            category="slotted_link",
            description=f"Oblong Slotted Link L={link_len:.0f} R={r_outer:.0f} slot={slot_len:.0f}x{r_slot*2:.0f}",
            entities=entities,
            relations=relations,
            annotations=annotations,
            dimensions=dimensions,
            cadcraft_commands=cadcraft_cmds,
            cadcraft_script="\n".join(scr_lines) + "\n",
            bounds=(0.0, 0.0, c_right[0] + r_outer, r_outer * 2),
        )

    @staticmethod
    def generate_keyway_cutout(sample_id: str, seed: Optional[int] = None) -> GeneratedSample:
        """Family 4: Shaft Hub / Collar with Keyway Notch Cutout."""
        rng = random.Random(seed)
        r_outer = rng.choice([40.0, 45.0, 50.0, 55.0])
        r_bore = rng.choice([20.0, 22.0, 25.0])
        kw_w = rng.choice([6.0, 8.0, 10.0])
        kw_depth = rng.choice([3.0, 4.0, 5.0])

        cx, cy = r_outer, r_outer
        center = [cx, cy]

        # Keyway slot at top (12 o'clock):
        # x goes from cx - kw_w/2 to cx + kw_w/2
        # y top goes to cy + r_bore + kw_depth
        x_left = cx - kw_w / 2.0
        x_right = cx + kw_w / 2.0
        y_top = cy + r_bore + kw_depth

        # Intersections of vertical slot sides with circle:
        # circle: center [cx,cy], radius r_bore
        # left line: x = x_left
        y_cut_l = cy + math.sqrt(max(r_bore**2 - (x_left - cx)**2, 0.0))
        y_cut_r = cy + math.sqrt(max(r_bore**2 - (x_right - cx)**2, 0.0))

        p_cut_l = [x_left, y_cut_l]
        p_cut_r = [x_right, y_cut_r]
        p_corner_l = [x_left, y_top]
        p_corner_r = [x_right, y_top]

        # Bore arc starts at p_cut_r and sweeps counter-clockwise around bottom to p_cut_l
        arc_info = arc_angles(center, p_cut_r, p_cut_l, [cx, cy - r_bore])

        entities = [
            # Outer circle
            CADEntity(id="e1", type="circle", center=center, radius=r_outer),
            # Keyway Left vertical line
            CADEntity(id="e2", type="line", p1=p_cut_l, p2=p_corner_l),
            # Keyway Top horizontal line
            CADEntity(id="e3", type="line", p1=p_corner_l, p2=p_corner_r),
            # Keyway Right vertical line
            CADEntity(id="e4", type="line", p1=p_corner_r, p2=p_cut_r),
            # Kept Bore Arc
            CADEntity(id="e5", type="arc", center=center, radius=r_bore, start_deg=arc_info['start_deg'], end_deg=arc_info['end_deg']),
        ]

        relations = [
            CADRelation(kind="coincident", refs=["e2", "e3"]),
            CADRelation(kind="coincident", refs=["e3", "e4"]),
            CADRelation(kind="coincident", refs=["e4", "e5"]),
            CADRelation(kind="coincident", refs=["e5", "e2"]),
            CADRelation(kind="radius", refs=["e1"], value_mm=r_outer),
            CADRelation(kind="radius", refs=["e5"], value_mm=r_bore),
        ]

        dimensions = [
            DimensionCallout(kind="diameter", p1=[], p2=[], label=f"Ø{r_outer*2:.1f}", center=center, radius=r_outer),
            DimensionCallout(kind="diameter", p1=[], p2=[], label=f"Ø{r_bore*2:.1f} BORE", center=center, radius=r_bore),
            DimensionCallout(kind="linear_h", p1=p_corner_l, p2=p_corner_r, label=f"W={kw_w:.1f}", offset=10.0),
            DimensionCallout(kind="linear_v", p1=[cx, cy + r_bore], p2=[cx, y_top], label=f"T={kw_depth:.1f}", offset=r_bore + 15.0),
        ]

        annotations = [
            CADAnnotation(text=f"HUB WITH KEYWAY CUTOUT {kw_w:.0f}x{kw_depth:.0f}", position=[10.0, r_outer * 2 + 15.0]),
        ]

        cadcraft_cmds = [
            {"cmd": "circle", "params": {"center": center, "radius": r_outer}},
            {"cmd": "line", "params": {"points": [p_cut_l, p_corner_l, p_corner_r, p_cut_r]}},
            {"cmd": "arc.ser", "params": {"start": p_cut_r, "end": p_cut_l, "radius": r_bore}},
            {"cmd": "dimradius", "params": {"center": center, "point": [center[0] + r_outer, center[1]]}},
            {"cmd": "dimlinear", "params": {"p1": p_corner_l, "p2": p_corner_r, "at": [cx, y_top + 10.0]}},
        ]

        scr_lines = [
            f"CIRCLE {center[0]},{center[1]} {r_outer}",
            f"LINE {p_cut_l[0]},{p_cut_l[1]} {p_corner_l[0]},{p_corner_l[1]} {p_corner_r[0]},{p_corner_r[1]} {p_cut_r[0]},{p_cut_r[1]} ",
            f"ARC C {center[0]},{center[1]} {p_cut_r[0]},{p_cut_r[1]} {p_cut_l[0]},{p_cut_l[1]}",
        ]

        return GeneratedSample(
            sample_id=sample_id,
            category="keyway_cutout",
            description=f"Shaft Hub Ro={r_outer:.0f} Ri={r_bore:.0f} with Keyway Cutout {kw_w:.0f}x{kw_depth:.0f}",
            entities=entities,
            relations=relations,
            annotations=annotations,
            dimensions=dimensions,
            cadcraft_commands=cadcraft_cmds,
            cadcraft_script="\n".join(scr_lines) + "\n",
            bounds=(0.0, 0.0, r_outer * 2, r_outer * 2 + kw_depth),
        )

    @staticmethod
    def generate_notched_flange(sample_id: str, seed: Optional[int] = None) -> GeneratedSample:
        """Family 5: Circular Flange with Circular Scallop / Notch Cutouts (Trim Joints)."""
        rng = random.Random(seed)
        r_outer = rng.choice([45.0, 50.0, 55.0, 60.0])
        r_notch = rng.choice([8.0, 10.0, 12.0])
        r_bore = rng.choice([15.0, 18.0, 20.0])
        num_notches = rng.choice([2, 4])

        cx, cy = r_outer, r_outer
        center = [cx, cy]

        # Notches positioned at 0, 90, 180, 270 degrees
        notch_angles = [i * (360.0 / num_notches) for i in range(num_notches)]
        entities = []
        relations = []

        # For 2 opposite notches at 0 and 180 deg:
        notch_centers = []
        for ang in notch_angles:
            rad = math.radians(ang)
            nc = [cx + r_outer * math.cos(rad), cy + r_outer * math.sin(rad)]
            notch_centers.append(nc)

        # Intersection points for each notch against the outer circle
        intersections = []
        for nc in notch_centers:
            res = circle_circle_intersections(center, r_outer, nc, r_notch)
            pts = res.get('points', [])
            intersections.append((nc, pts))

        # Build alternating outer arcs and inward notch arcs
        # For simplicity and watertightness in CADCraft, create the outer circle and inner bore,
        # with the notch cutout circles and radial drill hole pattern
        entities.append(CADEntity(id="e1", type="circle", center=center, radius=r_outer))
        entities.append(CADEntity(id="e2", type="circle", center=center, radius=r_bore))
        for i, nc in enumerate(notch_centers):
            entities.append(CADEntity(id=f"e_notch_{i+1}", type="circle", center=nc, radius=r_notch))
            relations.append(CADRelation(kind="coincident", refs=[f"e_notch_{i+1}", "e1"]))

        dimensions = [
            DimensionCallout(kind="diameter", p1=[], p2=[], label=f"Ø{r_outer*2:.1f} FLANGE", center=center, radius=r_outer),
            DimensionCallout(kind="diameter", p1=[], p2=[], label=f"Ø{r_bore*2:.1f} BORE", center=center, radius=r_bore),
            DimensionCallout(kind="radius", p1=[], p2=[], label=f"{num_notches}x R{r_notch:.1f} NOTCHES", center=notch_centers[0], radius=r_notch),
        ]

        annotations = [
            CADAnnotation(text=f"FLANGE WITH {num_notches} PERIMETER NOTCHES", position=[10.0, r_outer * 2 + 15.0]),
        ]

        cadcraft_cmds = [
            {"cmd": "circle", "params": {"center": center, "radius": r_outer}},
            {"cmd": "circle", "params": {"center": center, "radius": r_bore}},
        ]
        scr_lines = [
            f"CIRCLE {center[0]},{center[1]} {r_outer}",
            f"CIRCLE {center[0]},{center[1]} {r_bore}",
        ]
        for nc in notch_centers:
            cadcraft_cmds.append({"cmd": "circle", "params": {"center": nc, "radius": r_notch}})
            scr_lines.append(f"CIRCLE {nc[0]},{nc[1]} {r_notch}")

        cadcraft_cmds.append({"cmd": "dimradius", "params": {"center": center, "point": [center[0] + r_outer, center[1]]}})

        return GeneratedSample(
            sample_id=sample_id,
            category="notched_flange",
            description=f"Circular Flange Ro={r_outer:.0f} Ri={r_bore:.0f} with {num_notches} Notches R={r_notch:.0f}",
            entities=entities,
            relations=relations,
            annotations=annotations,
            dimensions=dimensions,
            cadcraft_commands=cadcraft_cmds,
            cadcraft_script="\n".join(scr_lines) + "\n",
            bounds=(0.0, 0.0, r_outer * 2, r_outer * 2),
        )


def generate_sample_by_index(idx: int, split: str = "train") -> GeneratedSample:
    """Deterministically generates a sample cycling through all 5 parametric families."""
    category_id = idx % 5
    sample_id = f"{split}_{idx:04d}"
    seed = idx + (100000 if split == "val" else 0)

    if category_id == 0:
        return ShapeGenerator.generate_corner_fillet_plate(sample_id, seed=seed)
    elif category_id == 1:
        return ShapeGenerator.generate_tangent_pulley(sample_id, seed=seed)
    elif category_id == 2:
        return ShapeGenerator.generate_slotted_link(sample_id, seed=seed)
    elif category_id == 3:
        return ShapeGenerator.generate_keyway_cutout(sample_id, seed=seed)
    else:
        return ShapeGenerator.generate_notched_flange(sample_id, seed=seed)


def execute_sample_in_cadcraft(sample: GeneratedSample, output_dir: str) -> Tuple[bool, str, str]:
    """Executes sample in cadcraft-cli to generate verified ground-truth DXF and rendered PNG.

    Returns:
        (success, dxf_path, png_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    dxf_path = os.path.join(output_dir, f"{sample.sample_id}.dxf")
    png_path = os.path.join(output_dir, f"{sample.sample_id}_cadcraft.png")

    cadcraft_cli = shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"
    if not os.path.exists(cadcraft_cli):
        return False, "", "cadcraft-cli executable not found"

    # Build command line arguments with --cmd chaining
    args = [cadcraft_cli, "run", "--metric"]
    for c in sample.cadcraft_commands:
        cmd_name = c["cmd"]
        params_str = json.dumps(c.get("params", {}))
        args.extend(["--cmd", f"{cmd_name} {params_str}"])

    args.extend(["--save", dxf_path, "--export", png_path])

    try:
        res = subprocess.run(args, capture_output=True, text=True, timeout=15)
        if res.returncode == 0 and os.path.exists(dxf_path):
            return True, dxf_path, png_path
        else:
            return False, dxf_path, f"CADCraft error: {res.stderr or res.stdout}"
    except Exception as e:
        return False, dxf_path, str(e)
