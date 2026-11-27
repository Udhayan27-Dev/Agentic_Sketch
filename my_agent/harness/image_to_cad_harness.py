"""Image-to-CAD Harness: Perception → Synthesis → Evaluation.

Sends the drawing to Qwen vision model with a clean, minimal prompt,
parses the JSON geometry spec, builds watertight CADCraft DXF commands,
executes headlessly, and audits via CADGeometryEvaluator.
"""

import argparse
import base64
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.evaluator import CADGeometryEvaluator, GeometricEvaluationReport
from sketchcad_agent.geometry_tools import (
    arc_angles,
    circle_circle_intersections,
    circle_line_intersections,
    close_joint,
    fillet_two_lines,
    line_line_intersection,
    tangent_arc_line_circle,
    tangent_from_point_to_circle,
    tangent_lines_two_circles,
    trim_extend_lines,
)

OUTPUT_DIR = PROJECT_ROOT.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_cadcraft_cli() -> str:
    env_bin = os.getenv("CADCRAFT_CLI_PATH")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin
    local = PROJECT_ROOT.parent / "cadcraft-main" / "target" / "debug" / "cadcraft-cli"
    if local.is_file() and os.access(str(local), os.X_OK):
        return str(local)
    return shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"


@dataclass
class ParsedDrawing:
    part_name: str
    part_type: str
    description: str
    bbox: Dict[str, float]
    outer_loop: List[Dict[str, Any]] = field(default_factory=list)
    holes: List[Dict[str, Any]] = field(default_factory=list)
    internal_lines: List[Dict[str, Any]] = field(default_factory=list)
    fillets: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class HarnessResult:
    image_path: str
    dxf_path: str
    png_path: str
    pdf_path: str
    parsed_drawing: ParsedDrawing
    evaluation: GeometricEvaluationReport
    cadcraft_commands: List[Dict[str, Any]]
    success: bool
    summary: str


# ---------------------------------------------------------------------------
#  Vision Model Query
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert mechanical CAD engineer. "
    "Analyze the 2D engineering drawing image and extract precise geometry. "
    "Output ONLY valid JSON — no markdown, no commentary."
)

ANALYSIS_PROMPT = """\
Analyze this 2D mechanical engineering drawing image and produce a JSON geometry specification.

RULES:
1. Coordinate origin (0,0): bottom-left of the part's bounding box. Y increases upward, X increases right. All units are mm.
2. outer_loop: ordered list of line and arc segments that form the CLOSED outer boundary of the part, traced counter-clockwise.
   - Line: {"id":"L1","type":"line","p1":[x1,y1],"p2":[x2,y2]}
   - Arc:  {"id":"A1","type":"arc","center":[cx,cy],"radius":r,"p_start":[sx,sy],"via_point":[mx,my],"p_end":[ex,ey]}
     via_point MUST be a point physically ON the arc midway between p_start and p_end (not on the opposite side of the circle).
3. holes: ALL circular features (boss rings AND bore holes), listed as {"center":[cx,cy],"radius":r}.
4. internal_lines: internal visible ribs as {"p1":[x1,y1],"p2":[x2,y2],"style":"solid"} or "dashed".
5. Read every printed dimension (Ø, R, linear) to set exact coordinates.
6. Never invent geometry not visible in the drawing.

Output ONLY this JSON:
{
  "part_name": "<snake_case_name>",
  "part_type": "<category>",
  "description": "<brief description>",
  "bbox": {"width": <W>, "height": <H>},
  "outer_loop": [...],
  "holes": [...],
  "internal_lines": [...]
}"""


class ImageCADHarness:
    """Automated Image-to-CAD pipeline: perception → synthesis → evaluation."""

    def __init__(self, cadcraft_cli: Optional[str] = None):
        self.cadcraft_cli = cadcraft_cli or find_cadcraft_cli()
        self.evaluator = CADGeometryEvaluator(tolerance_gap_mm=0.01, tolerance_tangent_deg=0.5)

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    def _prepare_image(self, image_path: str) -> Tuple[str, str]:
        """Load, optionally upscale/sharpen, return (base64, mime_type)."""
        from PIL import Image, ImageEnhance

        raw = Path(image_path).read_bytes()
        try:
            im = Image.open(io.BytesIO(raw))
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            w, h = im.size
            longest = max(w, h)
            if longest < 1300:
                scale = min(2.0, 1550.0 / max(1, longest))
                im = im.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
                im = ImageEnhance.Contrast(im).enhance(1.3)
                im = ImageEnhance.Sharpness(im).enhance(1.5)
            elif longest > 2048:
                im.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode(), "image/png"
        except Exception:
            ext = Path(image_path).suffix.lower().lstrip(".")
            mime = f"image/{ext}" if ext in ("png", "jpeg", "jpg", "webp") else "image/png"
            return base64.b64encode(raw).decode(), mime

    # ------------------------------------------------------------------
    # Vision model call (single turn — simple & reliable)
    # ------------------------------------------------------------------

    def query_vision_model(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Query Qwen with a clean single-turn prompt. No tools, no complexity."""
        api_base = (
            os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "http://127.0.0.1:8100/v1"
        )
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        model = os.getenv("SKETCHCAD_MODEL", "qwen3.8-flash-next").removeprefix("openai/")

        b64, mime = self._prepare_image(image_path)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYSIS_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            },
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            if resp.status_code != 200:
                print(f"[Vision] HTTP {resp.status_code}: {resp.text[:300]}", file=sys.stderr)
                return None
            text = resp.json()["choices"][0]["message"].get("content", "")
            return self._extract_json(text)
        except Exception as exc:
            print(f"[Vision] Query failed: {exc}", file=sys.stderr)
            return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract first valid JSON object from model text."""
        if not text:
            return None
        # Strip ```json ... ``` fences
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        raw = m.group(1).strip() if m else text.strip()
        # Try direct parse
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Find outermost braces
        start = raw.find("{")
        if start == -1:
            return None
        # Walk from last } backwards to find matching {
        depth = 0
        for i in range(len(raw) - 1, start - 1, -1):
            if raw[i] == "}":
                depth += 1
            elif raw[i] == "{":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except Exception:
                        break
        return None

    # ------------------------------------------------------------------
    # Image analysis → ParsedDrawing
    # ------------------------------------------------------------------

    def analyze_image(self, image_path: str) -> ParsedDrawing:
        """Run vision model and coerce output into a ParsedDrawing."""
        print("[PERCEPTION] Querying vision model...", file=sys.stderr)
        data = self.query_vision_model(image_path)
        print(f"[PERCEPTION] Raw response keys: {list(data.keys()) if data else 'None'}", file=sys.stderr)

        if not data or not isinstance(data, dict):
            data = {}

        # --- Part name ---
        raw_name = str(data.get("part_name") or data.get("part_type") or "mechanical_part")
        part_name = re.sub(r"[^a-zA-Z0-9]+", "_", raw_name.lower()).strip("_") or "mechanical_part"
        # Reject generic/bad names
        if part_name in ("out", "mechanical_part", "mechanical_component", "cad_part", "part"):
            part_name = "cad_part"

        # --- outer_loop ---
        outer_loop = []
        for ent in data.get("outer_loop") or []:
            if not isinstance(ent, dict):
                continue
            outer_loop.append(ent)

        # --- holes ---
        holes = []
        seen_holes = set()
        for h in data.get("holes") or []:
            c = h.get("center")
            r = float(h.get("radius", 0.0))
            if c and r > 0:
                key = (round(c[0], 1), round(c[1], 1), round(r, 2))
                if key not in seen_holes:
                    seen_holes.add(key)
                    holes.append(h)

        # --- internal_lines ---
        internal_lines = list(data.get("internal_lines") or [])

        bbox = data.get("bbox") or {}

        return ParsedDrawing(
            part_name=part_name,
            part_type=str(data.get("part_type", "Mechanical_Part")),
            description=str(data.get("description", "2D CAD Drawing")),
            bbox={"width": float(bbox.get("width", 100.0)), "height": float(bbox.get("height", 100.0))},
            outer_loop=outer_loop,
            holes=holes,
            internal_lines=internal_lines,
            raw_response=json.dumps(data),
        )

    # ------------------------------------------------------------------
    # Geometry synthesis → CADCraft commands
    # ------------------------------------------------------------------

    def _validate_arc_via_point(self, seg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures via_point lies on the correct (short) arc side between p1 and p2.
        Recomputes via_point from center+radius if the existing one implies a >180° sweep.
        """
        c = seg.get("center")
        r = seg.get("radius")
        if not c or not r:
            return seg
        p1 = seg.get("p1")
        p2 = seg.get("p2")
        via = seg.get("via")
        if not (p1 and p2 and via):
            return seg

        cx, cy = float(c[0]), float(c[1])
        r = float(r)

        def angle_of(pt):
            return math.atan2(pt[1] - cy, pt[0] - cx)

        a1 = math.degrees(angle_of(p1)) % 360
        a2 = math.degrees(angle_of(p2)) % 360
        av = math.degrees(angle_of(via)) % 360

        # CCW sweep from a1 to a2
        sweep_ccw = (a2 - a1) % 360
        # Angle of via relative to a1 (CCW)
        via_offset = (av - a1) % 360

        # If via is NOT between a1 and a2 (i.e., outside the CCW sweep), flip to opposite side
        if via_offset > sweep_ccw + 1e-6:
            # via is on the WRONG (long) side — recompute at the CCW midpoint
            mid_a = a1 + sweep_ccw / 2.0
            seg["via"] = [
                round(cx + r * math.cos(math.radians(mid_a)), 4),
                round(cy + r * math.sin(math.radians(mid_a)), 4),
            ]

        # If sweep > 350°, something is very wrong — recompute via at midpoint of short arc
        if sweep_ccw > 350:
            # Try the other direction (CW sweep = 360 - sweep_ccw)
            cw_sweep = 360 - sweep_ccw
            mid_a = a1 - cw_sweep / 2.0
            seg["via"] = [
                round(cx + r * math.cos(math.radians(mid_a)), 4),
                round(cy + r * math.sin(math.radians(mid_a)), 4),
            ]
        return seg

    def build_cadcraft_geometry(
        self, parsed: ParsedDrawing
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
        """Build verified CADCraft commands from the ParsedDrawing."""
        commands: List[Dict[str, Any]] = [
            {"cmd": "linetype", "params": {"load": "*"}},
            {"cmd": "layer.new", "params": {"name": "VISIBLE", "color": "white", "current": True}},
        ]
        gt_entities: List[Dict[str, Any]] = []

        # --- Outer loop ---
        loop_segments = []
        for idx, ent in enumerate(parsed.outer_loop):
            etype = str(ent.get("type", "")).lower()
            if etype == "line":
                p1 = ent.get("p1")
                p2 = ent.get("p2")
                if p1 and p2 and math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 1e-4:
                    loop_segments.append({
                        "type": "line",
                        "p1": [round(float(p1[0]), 4), round(float(p1[1]), 4)],
                        "p2": [round(float(p2[0]), 4), round(float(p2[1]), 4)],
                    })
            elif etype == "arc":
                c = ent.get("center")
                r = float(ent.get("radius", 0.0))
                if not c or r <= 0:
                    continue
                ps = ent.get("p_start") or ent.get("p1")
                pe = ent.get("p_end") or ent.get("p2")
                via = ent.get("via_point") or ent.get("via")
                s_deg = float(ent.get("start_deg", ent.get("start_angle", 0.0)))
                e_deg = float(ent.get("end_deg", ent.get("end_angle", 180.0)))
                if not ps:
                    ps = [c[0] + r * math.cos(math.radians(s_deg)), c[1] + r * math.sin(math.radians(s_deg))]
                if not pe:
                    pe = [c[0] + r * math.cos(math.radians(e_deg)), c[1] + r * math.sin(math.radians(e_deg))]
                if not via:
                    sweep = (e_deg - s_deg) % 360 or 180.0
                    m_deg = s_deg + sweep / 2.0
                    via = [c[0] + r * math.cos(math.radians(m_deg)), c[1] + r * math.sin(math.radians(m_deg))]
                seg = {
                    "type": "arc",
                    "p1": [round(float(ps[0]), 4), round(float(ps[1]), 4)],
                    "via": [round(float(via[0]), 4), round(float(via[1]), 4)],
                    "p2": [round(float(pe[0]), 4), round(float(pe[1]), 4)],
                    "center": c,
                    "radius": r,
                }
                seg = self._validate_arc_via_point(seg)
                loop_segments.append(seg)

        # Zero-gap snapping between adjacent segments
        n = len(loop_segments)
        for i in range(n):
            j = (i + 1) % n
            gap = math.hypot(
                loop_segments[j]["p1"][0] - loop_segments[i]["p2"][0],
                loop_segments[j]["p1"][1] - loop_segments[i]["p2"][1],
            )
            if 0 < gap <= 2.0:
                loop_segments[j]["p1"] = loop_segments[i]["p2"][:]

        for i, seg in enumerate(loop_segments):
            if seg["type"] == "line":
                commands.append({"cmd": "line", "params": {"points": [seg["p1"], seg["p2"]]}})
                gt_entities.append({"id": f"L{i+1}", "type": "line", "p1": seg["p1"], "p2": seg["p2"]})
            else:
                commands.append({"cmd": "arc", "params": {"p1": seg["p1"], "p2": seg["via"], "p3": seg["p2"]}})
                gt_entities.append({
                    "id": f"A{i+1}", "type": "arc",
                    "center": seg.get("center"), "radius": seg.get("radius"),
                    "p1": seg["p1"], "p2": seg["p2"],
                })

        contour_count = len(loop_segments)
        print(f"[SYNTHESIS] Outer loop: {contour_count} segments", file=sys.stderr)

        # --- Internal solid lines ---
        solid = [l for l in (parsed.internal_lines or []) if str(l.get("style", "solid")).lower() != "dashed"]
        if solid:
            commands.append({"cmd": "layer.new", "params": {"name": "INTERNAL", "color": "white", "current": True}})
            for ln in solid:
                p1, p2 = ln.get("p1"), ln.get("p2")
                if p1 and p2:
                    commands.append({"cmd": "line", "params": {"points": [
                        [round(float(p1[0]), 4), round(float(p1[1]), 4)],
                        [round(float(p2[0]), 4), round(float(p2[1]), 4)],
                    ]}})

        # --- Hidden dashed lines ---
        dashed = [l for l in (parsed.internal_lines or []) if str(l.get("style", "")).lower() == "dashed"]
        if dashed:
            commands.append({"cmd": "layer.new", "params": {"name": "HIDDEN", "color": "cyan", "linetype": "DASHED", "current": True}})
            for ln in dashed:
                p1, p2 = ln.get("p1"), ln.get("p2")
                if p1 and p2:
                    commands.append({"cmd": "line", "params": {"points": [
                        [round(float(p1[0]), 4), round(float(p1[1]), 4)],
                        [round(float(p2[0]), 4), round(float(p2[1]), 4)],
                    ]}})

        # --- Holes / Bores ---
        if parsed.holes:
            commands.append({"cmd": "layer.new", "params": {"name": "BORES", "color": "yellow", "current": True}})
            for h in parsed.holes:
                c = h.get("center")
                r = float(h.get("radius", 0.0))
                if c and r > 0:
                    commands.append({"cmd": "circle", "params": {"center": [round(c[0], 4), round(c[1], 4)], "radius": round(r, 4)}})
                    gt_entities.append({"id": f"bore_{len(gt_entities)+1}", "type": "circle", "center": c, "radius": r})

        # --- Centerlines ---
        has_cl = False
        for h in parsed.holes:
            c = h.get("center")
            r = float(h.get("radius", 0.0))
            if c and r > 0:
                if not has_cl:
                    commands.append({"cmd": "layer.new", "params": {"name": "CENTER", "color": "red", "current": True}})
                    has_cl = True
                ext = r + 6
                commands.append({"cmd": "line", "params": {"points": [[round(c[0]-ext,4), round(c[1],4)], [round(c[0]+ext,4), round(c[1],4)]]}})
                commands.append({"cmd": "line", "params": {"points": [[round(c[0],4), round(c[1]-ext,4)], [round(c[0],4), round(c[1]+ext,4)]]}})

        # --- Dimensions ---
        commands.append({"cmd": "layer.new", "params": {"name": "DIMENSIONS", "color": "green", "current": True}})
        w = float(parsed.bbox.get("width", 100.0))
        h = float(parsed.bbox.get("height", 60.0))
        commands.append({"cmd": "dimlinear", "params": {"p1": [0.0, 0.0], "p2": [w, 0.0], "at": [round(w/2, 2), -12.0]}})
        commands.append({"cmd": "dimlinear", "params": {"p1": [0.0, 0.0], "p2": [0.0, h], "at": [-12.0, round(h/2, 2)]}})

        gt_spec = {"units": "mm", "entities": gt_entities, "relations": []}
        desc = (
            f"Synthesized {parsed.part_type} with {contour_count} contour segments, "
            f"{len(parsed.holes)} holes/bores."
        )
        return commands, gt_spec, desc

    # ------------------------------------------------------------------
    # CADCraft execution
    # ------------------------------------------------------------------

    def execute_cadcraft(
        self, commands: List[Dict[str, Any]], output_dxf: str, output_png: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Execute commands headlessly via cadcraft-cli."""
        out_dxf = os.path.abspath(output_dxf)
        os.makedirs(os.path.dirname(out_dxf), exist_ok=True)

        args = [self.cadcraft_cli, "run", "--metric"]
        for cmd in commands:
            args.extend(["--cmd", f"{cmd['cmd']} {json.dumps(cmd.get('params', {}))}"])
        args.extend(["--save", out_dxf])
        if output_png:
            args.extend(["--export", os.path.abspath(output_png), "--view", "extents"])

        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=35)
            ok = res.returncode == 0 and os.path.isfile(out_dxf)
            return ok, (res.stdout or res.stderr).strip()
        except Exception as exc:
            return False, str(exc)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: str,
        output_dxf: Optional[str] = None,
        export_png: Optional[str] = None,
        open_gui: bool = False,
    ) -> HarnessResult:
        print("\n" + "=" * 70)
        print(f" SketchCAD Harness: {Path(image_path).name}")
        print("=" * 70)

        # 1. Perception
        parsed = self.analyze_image(image_path)
        print(f"[PERCEPTION] Part:        {parsed.part_name}")
        print(f"[PERCEPTION] Type:        {parsed.part_type}")
        print(f"[PERCEPTION] Bbox:        {parsed.bbox.get('width')} × {parsed.bbox.get('height')} mm")
        print(f"[PERCEPTION] Outer loop:  {len(parsed.outer_loop)} segments")
        print(f"[PERCEPTION] Holes:       {len(parsed.holes)}")

        # 2. Synthesis
        commands, gt_spec, desc = self.build_cadcraft_geometry(parsed)
        print(f"[SYNTHESIS] {desc}")
        print(f"[SYNTHESIS] {len(commands)} CADCraft commands generated.")

        # 3. Output paths
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", parsed.part_name.lower()).strip("_") or "cad_part"
        if slug in ("out", "cad_part_out"):
            slug = "cad_part"

        if output_dxf and output_dxf not in ("OUT.dxf",):
            dxf_path = os.path.abspath(output_dxf)
        else:
            dxf_path = str(OUTPUT_DIR / f"{slug}.dxf")

        out_dir = os.path.dirname(dxf_path)
        os.makedirs(out_dir, exist_ok=True)

        png_path = str(OUTPUT_DIR / f"{slug}_rendered.png") if not export_png else os.path.abspath(export_png)
        pdf_path = str(OUTPUT_DIR / f"{slug}.pdf")
        spec_path = str(OUTPUT_DIR / f"{slug}_vision_spec.json")

        try:
            Path(spec_path).write_text(parsed.raw_response)
        except Exception:
            pass

        # 4. Execute
        print(f"[CADCRAFT] Executing via {self.cadcraft_cli}...")
        ok, log = self.execute_cadcraft(commands, dxf_path, output_png=png_path)

        if not ok:
            print(f"[ERROR] CADCraft failed: {log[:300]}")
        else:
            print(f"[CADCRAFT] DXF: {dxf_path} ({os.path.getsize(dxf_path)} bytes)")
            if os.path.isfile(png_path):
                print(f"[CADCRAFT] PNG: {png_path} ({os.path.getsize(png_path)} bytes)")
            # Try PDF export
            try:
                subprocess.run([self.cadcraft_cli, "convert", dxf_path, pdf_path], capture_output=True, timeout=10)
            except Exception:
                pass

        # 5. Geometric audit
        eval_report = self.evaluator.evaluate_dxf(dxf_path, gt_spec, sample_id=slug)
        print("-" * 70)
        print(f"[AUDIT] {eval_report.summary}")
        print(f"[AUDIT] Watertight: {eval_report.is_watertight}  Max gap: {eval_report.max_gap_mm:.4f} mm")
        print(f"[AUDIT] Tangency error: {eval_report.tangency_mean_error_deg:.4f}°")
        print(f"[AUDIT] Accuracy: {eval_report.accuracy_score:.1f}%")
        print("-" * 70)

        # 6. Optional GUI
        if open_gui and ok:
            from sketchcad_agent.agent import open_in_cadcraft_gui
            msg = open_in_cadcraft_gui(dxf_path)
            print(f"[GUI] {msg}")

        summary = (
            f"Converted {Path(image_path).name} → {Path(dxf_path).name} "
            f"({eval_report.accuracy_score:.1f}% accuracy, "
            f"{eval_report.max_gap_mm:.4f}mm gap, "
            f"{eval_report.tangency_mean_error_deg:.2f}° tangency)"
        )

        return HarnessResult(
            image_path=image_path,
            dxf_path=dxf_path,
            png_path=png_path,
            pdf_path=pdf_path,
            parsed_drawing=parsed,
            evaluation=eval_report,
            cadcraft_commands=commands,
            success=ok and eval_report.is_watertight and eval_report.tangency_mean_error_deg < 0.5,
            summary=summary,
        )


def main():
    parser = argparse.ArgumentParser(description="Image-to-CAD Harness")
    parser.add_argument("--image", required=True, help="Path to drawing image")
    parser.add_argument("--out", default=None, help="Output DXF path (default: /home/udhayan/Agent/outputs/<name>.dxf)")
    parser.add_argument("--export", default=None, help="Output PNG path")
    parser.add_argument("--gui", action="store_true", help="Open CADCraft GUI on completion")
    args = parser.parse_args()

    harness = ImageCADHarness()
    result = harness.process_image(
        image_path=args.image,
        output_dxf=args.out,
        export_png=args.export,
        open_gui=args.gui,
    )
    print(f"\n[DONE] {result.summary}")
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
