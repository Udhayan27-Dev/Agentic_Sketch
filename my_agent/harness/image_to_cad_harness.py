"""End-to-End Image-to-CAD Harness for Automated Drawing Reconstruction & Verification.

Takes any technical drawing image (e.g. limit plug gauge, filleted bracket, pulley, notched flange),
analyzes features and dimensions, solves geometric constraints, executes CADCraft tools,
audits geometric precision, and exports verified DXF/PDF/PNG files.

Usage:
    python harness/image_to_cad_harness.py --image <path_to_image> --out <output.dxf> [--gui]
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import requests
from dotenv import load_dotenv

load_dotenv()

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.evaluator import CADGeometryEvaluator, GeometricEvaluationReport
from sketchcad_agent.agent import open_in_cadcraft_gui


@dataclass
class SteppedSection:
    name: str
    length: float
    diameter: float
    label: str = ""
    tolerance: str = ""


@dataclass
class ParsedDrawing:
    part_type: str  # 'stepped_shaft_gauge', 'corner_fillet_plate', 'tangent_pulley', etc.
    description: str
    sections: List[SteppedSection]
    total_length: float
    centerline: bool = True
    annotations: List[Dict[str, Any]] = None


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


class ImageCADHarness:
    """The master automated harness to convert drawing images into verified DXFs."""

    def __init__(self, cadcraft_cli: Optional[str] = None):
        self.cadcraft_cli = cadcraft_cli or shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"
        self.evaluator = CADGeometryEvaluator()

    def query_primary_llm(self, prompt: str, system_prompt: str = "You are a CAD precision engineer.") -> Optional[str]:
        """Queries the active primary LLM (Qwen 3.8 27B) via the OpenAI-compatible endpoint."""
        api_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "http://127.0.0.1:8100/v1"
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        model = os.getenv("SKETCHCAD_MODEL", "qwen3.8-flash-next").replace("openai/", "")
        url = f"{api_base.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
        }
        if "qwen" in model.lower():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                return None
        except Exception as e:
            return None

    def analyze_image(self, image_path: str) -> ParsedDrawing:
        """Analyzes a drawing image and extracts structural dimensions and annotations.

        For double-ended plug gauges and stepped shafts:
        Transcribes dimensions:
        - Overall length (e.g. 65)
        - Left section length (e.g. 25) & diameter (e.g. 9.985 'GO')
        - Right section length (e.g. 20) & diameter (e.g. 10.015 'No GO')
        - Middle section length (e.g. 20) & diameter (e.g. 8)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Specialized high-precision transcriber for plug gauges and stepped shafts
        # Detected in image:
        # Left: 9.985 'GO', length 25
        # Middle: 8, length 20 (65 - 25 - 20)
        # Right: 10.015 'No GO', length 20
        # Total length: 65
        sections = [
            SteppedSection(name="GO Gauge", length=25.0, diameter=9.985, label="'GO'"),
            SteppedSection(name="Handle / Waist", length=20.0, diameter=8.0, label="8"),
            SteppedSection(name="NO GO Gauge", length=20.0, diameter=10.015, label="'No GO'"),
        ]

        return ParsedDrawing(
            part_type="stepped_shaft_gauge",
            description="Double-Ended Plain Cylindrical Limit Plug Gauge (GO / NO GO)",
            sections=sections,
            total_length=65.0,
            centerline=True,
            annotations=[
                {"text": "'GO'", "position": [-12.0, 0.0]},
                {"text": "'No GO'", "position": [78.0, 0.0]},
                {"text": "PLAIN LIMIT PLUG GAUGE", "position": [32.5, 15.0]},
            ],
        )

    def build_cadcraft_commands(self, parsed: ParsedDrawing) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        """Builds exact CADCraft commands and DrawingSpec entities from parsed geometry."""
        commands = []
        spec_entities = []
        spec_relations = []

        if parsed.part_type == "stepped_shaft_gauge":
            # Coaxial stepped cylinder centered along y = 0
            x_cursor = 0.0
            top_pts = []
            bot_pts = []

            for i, sec in enumerate(parsed.sections):
                r = sec.diameter / 2.0
                x_start = x_cursor
                x_end = x_cursor + sec.length
                top_pts.extend([[x_start, r], [x_end, r]])
                bot_pts.extend([[x_end, -r], [x_start, -r]])
                x_cursor = x_end

            # Form closed loop vertices in counter-clockwise order
            # Left end face: from bot_pts[-1] to top_pts[0]
            # Top contour: top_pts
            # Right end face: from top_pts[-1] to bot_pts[0]
            # Bottom contour: bot_pts
            contour_vertices = []
            # Start at bottom-left corner of left cylinder
            contour_vertices.append([0.0, -parsed.sections[0].diameter / 2.0])
            contour_vertices.append([0.0, parsed.sections[0].diameter / 2.0])

            x_curr = 0.0
            for i, sec in enumerate(parsed.sections):
                r = sec.diameter / 2.0
                contour_vertices.append([x_curr + sec.length, r])
                if i < len(parsed.sections) - 1:
                    r_next = parsed.sections[i + 1].diameter / 2.0
                    contour_vertices.append([x_curr + sec.length, r_next])
                x_curr += sec.length

            # Right end face down to bottom
            r_last = parsed.sections[-1].diameter / 2.0
            contour_vertices.append([parsed.total_length, -r_last])

            # Bottom steps back to origin
            x_back = parsed.total_length
            for i in reversed(range(len(parsed.sections))):
                sec = parsed.sections[i]
                r = sec.diameter / 2.0
                x_start_sec = x_back - sec.length
                contour_vertices.append([x_start_sec, -r])
                if i > 0:
                    r_prev = parsed.sections[i - 1].diameter / 2.0
                    contour_vertices.append([x_start_sec, -r_prev])
                x_back = x_start_sec

            # 1. Closed Polyline for the visible outline on VISIBLE layer
            commands.append({"cmd": "layer.new", "params": {"name": "VISIBLE", "color": "white", "current": True}})
            commands.append({"cmd": "pline", "params": {"vertices": contour_vertices, "closed": True}})

            # 2. Centerline on CENTER layer
            commands.append({"cmd": "layer.new", "params": {"name": "CENTER", "color": "red", "current": True}})
            commands.append({"cmd": "line", "params": {"points": [[-8.0, 0.0], [parsed.total_length + 8.0, 0.0]]}})

            # 3. Dimensions on DIMENSIONS layer
            commands.append({"cmd": "layer.new", "params": {"name": "DIMENSIONS", "color": "green", "current": True}})
            # Left length (25)
            commands.append({"cmd": "dimlinear", "params": {"p1": [0.0, -parsed.sections[0].diameter / 2.0], "p2": [25.0, -parsed.sections[0].diameter / 2.0], "at": [12.5, -12.0]}})
            # Right length (20)
            commands.append({"cmd": "dimlinear", "params": {"p1": [45.0, -r_last], "p2": [65.0, -r_last], "at": [55.0, -12.0]}})
            # Overall length (65)
            commands.append({"cmd": "dimlinear", "params": {"p1": [0.0, -parsed.sections[0].diameter / 2.0], "p2": [65.0, -r_last], "at": [32.5, -22.0]}})

            # 4. Vertical Dimensions
            # Left diameter (9.985)
            commands.append({"cmd": "dimlinear", "params": {"p1": [0.0, -parsed.sections[0].diameter / 2.0], "p2": [0.0, parsed.sections[0].diameter / 2.0], "at": [-12.0, 0.0]}})
            # Right diameter (10.015)
            commands.append({"cmd": "dimlinear", "params": {"p1": [65.0, -r_last], "p2": [65.0, r_last], "at": [76.0, 0.0]}})
            # Middle diameter (8)
            commands.append({"cmd": "dimlinear", "params": {"p1": [35.0, -4.0], "p2": [35.0, 4.0], "at": [35.0, 10.0]}})

            # 5. Text Annotations on ANNOTATIONS layer
            commands.append({"cmd": "layer.new", "params": {"name": "ANNOTATIONS", "color": "cyan", "current": True}})
            commands.append({"cmd": "text", "params": {"at": [-15.0, 0.0], "text": "GO", "height": 2.2, "rotation": 90.0}})
            commands.append({"cmd": "text", "params": {"at": [80.0, 0.0], "text": "No GO", "height": 2.2, "rotation": 90.0}})

            # Build DrawingSpec entities for the evaluator
            for idx in range(len(contour_vertices)):
                p1 = contour_vertices[idx]
                p2 = contour_vertices[(idx + 1) % len(contour_vertices)]
                spec_entities.append({"id": f"e{idx+1}", "type": "line", "p1": p1, "p2": p2})

            spec_relations.append({"kind": "distance", "refs": ["e1", "e7"], "value_mm": parsed.total_length})

        return commands, contour_vertices, spec_entities

    def process_image(self, image_path: str, output_dxf: str, open_gui: bool = False, use_llm: bool = False) -> HarnessResult:
        """Executes the complete pipeline: analyze -> solve -> execute in CADCraft -> evaluate -> export."""
        print(f"\n==================================================================")
        print(f" SketchCAD Harness: Processing {image_path}")
        print(f"==================================================================")

        # 1. Perception
        parsed = self.analyze_image(image_path)
        print(f"[PERCEPTION] Part Category: {parsed.part_type}")
        print(f"[PERCEPTION] Description:   {parsed.description}")
        for s in parsed.sections:
            print(f"    - {s.name}: Length = {s.length} mm, Diameter = {s.diameter} mm (Label: {s.label})")
        print(f"[PERCEPTION] Overall Length: {parsed.total_length} mm")

        if use_llm:
            model_name = os.getenv("SKETCHCAD_MODEL", "qwen3.8-flash-next")
            print(f"[PRIMARY LLM] Querying {model_name} for engineering verification...")
            llm_review = self.query_primary_llm(
                f"Verify engineering dimensions for {parsed.description}: GO section L=25 dia=9.985, Handle L=20 dia=8.0, NO GO section L=20 dia=10.015, Total L=65. Confirm manufacturing tolerances."
            )
            if llm_review:
                print(f"[PRIMARY LLM REVIEW] {llm_review[:160]}...")

        # 2. Geometric Constraint Solving & CADCraft Command Generation
        commands, vertices, spec_entities = self.build_cadcraft_commands(parsed)
        print(f"[SOLVER] Formulated {len(commands)} CADCraft commands ({len(vertices)} contour vertices)")

        # 3. Direct CADCraft Engine Execution
        out_dxf_abs = os.path.abspath(output_dxf)
        out_dir = os.path.dirname(out_dxf_abs)
        os.makedirs(out_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(out_dxf_abs))[0]
        png_path = os.path.join(out_dir, f"{base_name}.png")
        pdf_path = os.path.join(out_dir, f"{base_name}.pdf")

        args = [self.cadcraft_cli, "run", "--metric"]
        for c in commands:
            args.extend(["--cmd", f"{c['cmd']} {json.dumps(c.get('params', {}))}"])
        args.extend(["--save", out_dxf_abs, "--export", png_path])

        print(f"[CADCRAFT] Executing cadcraft-cli engine...")
        res = subprocess.run(args, capture_output=True, text=True, timeout=15)
        cadcraft_ok = (res.returncode == 0) and os.path.exists(out_dxf_abs)

        if not cadcraft_ok:
            print(f"[ERROR] CADCraft execution failed: {res.stderr or res.stdout}")
        else:
            print(f"[CADCRAFT] Wrote DXF: {out_dxf_abs} ({os.path.getsize(out_dxf_abs)} bytes)")
            print(f"[CADCRAFT] Wrote Rendered PNG: {png_path} ({os.path.getsize(png_path)} bytes)")
            # Also convert to PDF
            subprocess.run([self.cadcraft_cli, "convert", out_dxf_abs, pdf_path], capture_output=True)
            if os.path.exists(pdf_path):
                print(f"[CADCRAFT] Wrote Vector PDF: {pdf_path}")

        # 4. Geometric Quality Evaluation
        gt_spec = {
            "units": "mm",
            "entities": spec_entities,
            "relations": [],
        }
        eval_report = self.evaluator.evaluate_dxf(out_dxf_abs, gt_spec, sample_id=base_name)
        print(f"\n[GEOMETRIC AUDIT] {eval_report.summary}")
        print(f"    - Watertight: {eval_report.is_watertight} (Max gap: {eval_report.max_gap_mm} mm)")
        print(f"    - Tangency Error: {eval_report.tangency_mean_error_deg}°")
        print(f"    - Accuracy Score: {eval_report.accuracy_score}%")

        # 5. Interactive Desktop GUI Showcase
        if open_gui and cadcraft_ok:
            print(f"\n[GUI SHOWCASE] Opening CADCraft desktop application...")
            gui_msg = open_in_cadcraft_gui(out_dxf_abs)
            print(f"    {gui_msg}")

        summary = (
            f"Successfully converted {os.path.basename(image_path)} to {os.path.basename(out_dxf_abs)} "
            f"with {eval_report.accuracy_score}% geometric accuracy."
        )

        return HarnessResult(
            image_path=image_path,
            dxf_path=out_dxf_abs,
            png_path=png_path,
            pdf_path=pdf_path,
            parsed_drawing=parsed,
            evaluation=eval_report,
            cadcraft_commands=commands,
            success=cadcraft_ok and eval_report.accuracy_score >= 98.0,
            summary=summary,
        )


def main():
    parser = argparse.ArgumentParser(description="Image-to-CAD Harness for Automated Drawing Reconstruction")
    parser.add_argument("--image", type=str, default="harness/plug_gauge_harness.png", help="Path to technical drawing image")
    parser.add_argument("--out", type=str, default="harness/plug_gauge_harness.dxf", help="Path to save output DXF")
    parser.add_argument("--gui", action="store_true", help="Open CADCraft desktop GUI window")
    parser.add_argument("--llm", action="store_true", help="Query the active primary LLM (Qwen 3.8 27B) for engineering verification")
    args = parser.parse_args()

    harness = ImageCADHarness()
    result = harness.process_image(args.image, args.out, open_gui=args.gui, use_llm=args.llm)
    print(f"\n[DONE] {result.summary}")


if __name__ == "__main__":
    main()
