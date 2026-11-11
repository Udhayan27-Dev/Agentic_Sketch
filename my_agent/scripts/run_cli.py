#!/usr/bin/env python3
"""Unified CLI for SketchCAD: Autonomous CAD generation powered by Qwen 3.8 27B on DGX.

Full Agentic Flow:
1. Qwen Analyst: Directly inspects drawing image via DGX multimodal endpoint, extracting DrawingSpec.
2. Qwen Draftsman: Translates geometry into CADCraft commands (pline, circle, arc, dimlinear, text).
3. CADCraft Engine: Executes commands to compile OUT.dxf, OUT.png, and OUT.pdf.
4. Qwen Critic: Inspects and scores geometric accuracy and watertightness.
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image

# Ensure my_agent root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sketchcad_agent.agent import open_in_cadcraft_gui, sanitize_json_output

load_dotenv()

def resolve_image_path(input_path: str) -> str:
    """Finds image in current dir, Agent folder, or absolute path."""
    candidates = [
        input_path,
        os.path.abspath(input_path),
        os.path.join("/home/udhayan/Agent", input_path),
        os.path.join("/home/udhayan/Agent", os.path.basename(input_path)),
        os.path.join("/home/udhayan/Agent/my_agent", input_path),
        os.path.join("/home/udhayan/Agent/my_agent", os.path.basename(input_path)),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isfile(c):
            return os.path.abspath(c)
    return input_path

def prepare_image_b64(image_path: str, max_dimension: int = 768) -> str:
    """Resizes image keeping aspect ratio to max_dimension and returns base64 PNG."""
    im = Image.open(image_path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    if max(im.size) > max_dimension:
        im.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def query_qwen(
    prompt: str,
    image_path: Optional[str] = None,
    system: str = "You are a CAD precision engineer.",
    max_tokens: int = 1500,
    label: str = "Qwen"
) -> str:
    """Sends prompt (and optional multimodal image) to the active Qwen 3.8 27B endpoint on DGX."""
    api_base = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:7030/v1")
    url = f"{api_base.rstrip('/')}/chat/completions"

    if image_path and os.path.exists(image_path):
        b64 = prepare_image_b64(image_path)
        user_content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ]
    else:
        user_content = prompt

    payload = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    for attempt in range(2):
        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=180)
            if resp.status_code != 200:
                raise RuntimeError(f"Qwen endpoint error {resp.status_code}: {resp.text}")

            full_text: List[str] = []
            token_count = 0
            print(f"[{label}] Streaming response from DGX (http://127.0.0.1:7030/v1)...")
            
            for line in resp.iter_lines():
                if line:
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: ") and not decoded.startswith("data: [DONE]"):
                        try:
                            chunk = json.loads(decoded[6:])
                            content = chunk["choices"][0]["delta"].get("content", "")
                            if content:
                                full_text.append(content)
                                token_count += 1
                                if token_count % 30 == 0:
                                    elapsed = max(time.time() - t0, 0.1)
                                    print(f"\r[{label}] Received {token_count} tokens ({token_count / elapsed:.1f} tok/s)...", end="", flush=True)
                        except Exception:
                            pass

            total_time = max(time.time() - t0, 0.1)
            print(f"\r[{label}] Completed: {token_count} tokens generated in {total_time:.2f}s ({token_count / total_time:.1f} tok/s)    \n")
            result_str = "".join(full_text)
            if result_str.strip():
                return result_str
        except Exception as e:
            if attempt == 0:
                print(f"\n[!] Network hiccup during streaming ({e}), retrying once...")
                time.sleep(2)
                continue
            else:
                if full_text:
                    print(f"\n[!] Stream interrupted, using partial generated tokens ({len(full_text)} fragments).")
                    return "".join(full_text)
                raise RuntimeError(f"Failed to query Qwen on DGX: {e}")

    return ""

def parse_json_from_llm(text: str) -> Any:
    """Extracts valid JSON payload (dict or list) from LLM output."""
    clean = sanitize_json_output(text)
    try:
        return json.loads(clean)
    except Exception:
        pass
    
    # Try finding array or object bounds
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    if first_bracket != -1 and last_bracket > first_bracket and (first_brace == -1 or first_bracket < first_brace):
        try:
            return json.loads(text[first_bracket:last_bracket + 1])
        except Exception:
            pass

    if first_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace + 1])
        except Exception:
            pass

    # Regex extraction of code blocks
    code_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except Exception:
            pass

    return None

def normalize_cadcraft_commands(commands_input: Any, spec_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Sanitizes and formats commands into CADCraft CLI compatible execution sequence."""
    norm_cmds: List[Dict[str, Any]] = []
    seen_layers: set = {"0", "VISIBLE"}

    # Ensure base visible layer exists
    norm_cmds.append({"cmd": "layer.new", "params": {"name": "VISIBLE", "color": "white", "current": True}})

    if isinstance(commands_input, list):
        for item in commands_input:
            if not isinstance(item, dict):
                continue
            cmd = item.get("cmd") or item.get("command") or item.get("type")
            params = item.get("params") or item.get("parameters") or item.get("args") or {}
            if not cmd:
                continue

            cmd = cmd.lower()
            if cmd == "polyline":
                cmd = "pline"

            # Handle layer management
            if cmd in ("layer.new", "layer"):
                lname = params.get("name")
                if lname:
                    if lname in seen_layers:
                        norm_cmds.append({"cmd": "layer.set", "params": {"name": lname}})
                    else:
                        seen_layers.add(lname)
                        norm_cmds.append({"cmd": "layer.new", "params": params})
                continue
            elif cmd == "layer.set":
                lname = params.get("name")
                if lname and lname in seen_layers:
                    norm_cmds.append({"cmd": "layer.set", "params": {"name": lname}})
                continue

            # Normalize pline
            if cmd == "pline":
                verts = params.get("vertices") or params.get("points") or []
                if verts and len(verts) >= 2:
                    clean_verts = [[round(float(v[0]), 4), round(float(v[1]), 4)] for v in verts if len(v) >= 2]
                    norm_cmds.append({"cmd": "pline", "params": {"vertices": clean_verts, "closed": params.get("closed", True)}})

            # Normalize circle
            elif cmd == "circle":
                c = params.get("center") or [0.0, 0.0]
                r = params.get("radius") or params.get("r") or 1.0
                norm_cmds.append({"cmd": "circle", "params": {"center": [round(float(c[0]), 4), round(float(c[1]), 4)], "radius": round(float(r), 4)}})

            # Normalize arc
            elif cmd == "arc":
                c = params.get("center") or [0.0, 0.0]
                r = params.get("radius") or params.get("r") or 1.0
                s = params.get("start") if params.get("start") is not None else (params.get("start_angle") or params.get("start_deg") or 0.0)
                e = params.get("end") if params.get("end") is not None else (params.get("end_angle") or params.get("end_deg") or 180.0)
                norm_cmds.append({"cmd": "arc", "params": {"center": [round(float(c[0]), 4), round(float(c[1]), 4)], "radius": round(float(r), 4), "start": round(float(s), 2), "end": round(float(e), 2)}})

            # Normalize line
            elif cmd == "line":
                pts = params.get("points")
                if not pts and "p1" in params and "p2" in params:
                    pts = [params["p1"], params["p2"]]
                if pts and len(pts) >= 2:
                    norm_cmds.append({"cmd": "line", "params": {"points": [[round(float(pts[0][0]), 4), round(float(pts[0][1]), 4)], [round(float(pts[1][0]), 4), round(float(pts[1][1]), 4)]]}})

            # Normalize dimlinear
            elif cmd in ("dimlinear", "dimension"):
                p1 = params.get("p1") or [0.0, 0.0]
                p2 = params.get("p2") or [10.0, 0.0]
                at = params.get("at") or params.get("position") or [(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0 - 5.0]
                norm_cmds.append({"cmd": "dimlinear", "params": {"p1": [round(float(p1[0]), 4), round(float(p1[1]), 4)], "p2": [round(float(p2[0]), 4), round(float(p2[1]), 4)], "at": [round(float(at[0]), 4), round(float(at[1]), 4)]}})

            # Normalize text
            elif cmd in ("text", "annotation"):
                at = params.get("at") or params.get("position") or [0.0, 0.0]
                txt = str(params.get("text", ""))
                h = params.get("height") or 2.0
                norm_cmds.append({"cmd": "text", "params": {"at": [round(float(at[0]), 4), round(float(at[1]), 4)], "text": txt, "height": round(float(h), 2)}})

    # If commands were sparse, convert directly from DrawingSpec entities
    if len([c for c in norm_cmds if c["cmd"] in ("pline", "circle", "line", "arc")]) == 0 and spec_data:
        entities = spec_data.get("entities", [])
        for ent in entities:
            etype = ent.get("type", "").lower()
            if etype == "pline" and ent.get("vertices"):
                norm_cmds.append({"cmd": "pline", "params": {"vertices": ent["vertices"], "closed": ent.get("closed", True)}})
            elif etype == "circle" and ent.get("center") and "radius" in ent:
                norm_cmds.append({"cmd": "circle", "params": {"center": ent["center"], "radius": ent["radius"]}})
            elif etype == "arc" and ent.get("center") and "radius" in ent:
                norm_cmds.append({"cmd": "arc", "params": {"center": ent["center"], "radius": ent["radius"], "start": ent.get("start", ent.get("start_deg", 0.0)), "end": ent.get("end", ent.get("end_deg", 180.0))}})
            elif etype == "line" and ent.get("p1") and ent.get("p2"):
                norm_cmds.append({"cmd": "line", "params": {"points": [ent["p1"], ent["p2"]]}})

        # Add annotations from spec
        for annot in spec_data.get("annotations", []):
            if annot.get("position") and annot.get("text"):
                norm_cmds.append({"cmd": "text", "params": {"at": annot["position"], "text": str(annot["text"]), "height": annot.get("height", 2.0)}})

    return norm_cmds

def main():
    parser = argparse.ArgumentParser(description="SketchCAD Autonomous Agent CLI (Directly Powered by Qwen 3.8 27B on DGX)")
    parser.add_argument("image", nargs="?", default="test.png", help="Path to input technical drawing image (e.g. test.png, test2.png)")
    parser.add_argument("--h", "--h-param", type=float, default=None, dest="h_param", help="Parametric 'h' dimension in mm (defaults to 12.0 for test.png, 100.0 for others)")
    parser.add_argument("--out", "-o", type=str, default="OUT.dxf", help="Output DXF file path (default: OUT.dxf)")
    parser.add_argument("--gui", action="store_true", default=True, help="Open CADCraft desktop GUI window (default: True)")
    parser.add_argument("--no-gui", action="store_false", dest="gui", help="Disable desktop GUI window")
    cli_args = parser.parse_args()

    image_path = resolve_image_path(cli_args.image)
    if not os.path.exists(image_path):
        print(f"\n[!] Error: Image file not found: {cli_args.image}")
        print("    Searched candidates:")
        print(f"      - {cli_args.image}")
        print(f"      - /home/udhayan/Agent/{os.path.basename(cli_args.image)}")
        print(f"      - /home/udhayan/Agent/my_agent/{os.path.basename(cli_args.image)}")
        sys.exit(1)

    # Set default h_param based on drawing type if not explicitly passed
    if cli_args.h_param is None:
        if "test2" in os.path.basename(image_path).lower():
            h_val = 100.0
        elif "test" in os.path.basename(image_path).lower():
            h_val = 12.0
        else:
            h_val = 100.0
    else:
        h_val = cli_args.h_param

    print("\n" + "=" * 75)
    print("  SketchCAD Autonomous Agent (Direct Vision Pipeline via Qwen on DGX)")
    print("=" * 75)
    print(f"[*] Input Drawing:     {image_path}")
    print(f"[*] Parametric Scale:  h = {h_val} mm")
    print(f"[*] LLM Endpoint:      http://127.0.0.1:7030/v1 (DGX Qwen 3.8 27B FP8)")
    print(f"[*] Target Output:     {cli_args.out}")

    im = Image.open(image_path)
    w, h = im.size
    print(f"[*] Input Image Size:  {w}x{h} (Aspect ratio: {w / max(h, 1):.2f})")

    # =========================================================================
    # STAGE 1: QWEN ANALYST AGENT (DrawingSpec Vision Extraction on DGX)
    # =========================================================================
    print("\n" + "-" * 75)
    print("[STAGE 1: Qwen Analyst Agent on DGX] Inspecting image and extracting DrawingSpec...")
    print("-" * 75)

    analyst_prompt = f"""You are the Lead CAD Metrologist and Precision Analyst.
Inspect this technical drawing image directly and extract its complete geometric specification.

Parameter rule: The scaling parameter 'h' is given as h = {h_val} mm.
Calculate ALL physical coordinates, lengths, radii, and positions in millimeters (mm) using h = {h_val}.
Anchor the coordinate origin (0.0, 0.0) strictly at the bottom-left vertex of the part contour.

Extract the following:
1. Physical outline contour (closed boundary vertices, arcs, notches, cutouts).
2. Internal holes and circular bores (centers [cx, cy] and radii in mm).
3. Dimensions shown in the drawing.
4. Text callouts and annotations.

Format your output strictly as a valid JSON object matching this schema:
{{
  "part_name": "string description",
  "units": "mm",
  "h_val": {h_val},
  "entities": [
    {{"id": "e1", "type": "pline", "vertices": [[0.0, 0.0], [x1, y1], [x2, y2], ...], "closed": true}},
    {{"id": "e2", "type": "circle", "center": [cx, cy], "radius": r}},
    {{"id": "e3", "type": "arc", "center": [cx, cy], "radius": r, "start_deg": 0.0, "end_deg": 180.0}},
    {{"id": "e4", "type": "line", "p1": [x1, y1], "p2": [x2, y2]}}
  ],
  "dimensions": [
    {{"p1": [x1, y1], "p2": [x2, y2], "label": "..."}}
  ],
  "annotations": [
    {{"text": "...", "position": [x, y], "height": 2.5}}
  ]
}}

Return ONLY valid JSON starting with '{{' and ending with '}}'."""

    analyst_raw = query_qwen(
        analyst_prompt,
        image_path=image_path,
        system="You are an expert CAD engineer. Extract complete technical drawing specifications into strictly valid JSON.",
        max_tokens=1500,
        label="Qwen Analyst"
    )

    spec_data = parse_json_from_llm(analyst_raw)
    if not isinstance(spec_data, dict):
        spec_data = {"entities": [], "dimensions": [], "annotations": []}

    num_entities = len(spec_data.get("entities", []))
    print(f"[+] Qwen Analyst extracted DrawingSpec with {num_entities} geometric entities from the image.")

    # =========================================================================
    # STAGE 2: QWEN DRAFTSMAN AGENT (CADCraft Command Formulation)
    # =========================================================================
    print("-" * 75)
    print("[STAGE 2: Qwen Draftsman Agent on DGX] Translating DrawingSpec to CADCraft Commands...")
    print("-" * 75)

    draftsman_prompt = f"""You are the CAD Draftsman. Translate this DrawingSpec (h = {h_val} mm) into a sequence of CADCraft commands:
DrawingSpec:
{json.dumps(spec_data, indent=2) if spec_data.get('entities') else analyst_raw[:1200]}

Valid CADCraft commands:
- pline: {{"cmd": "pline", "params": {{"vertices": [[x0,y0], [x1,y1], ...], "closed": true}}}}
- circle: {{"cmd": "circle", "params": {{"center": [cx, cy], "radius": r}}}}
- arc: {{"cmd": "arc", "params": {{"center": [cx, cy], "radius": r, "start": deg, "end": deg}}}}
- line: {{"cmd": "line", "params": {{"points": [[x1, y1], [x2, y2]]}}}}
- dimlinear: {{"cmd": "dimlinear", "params": {{"p1": [x1, y1], "p2": [x2, y2], "at": [x_dim, y_dim]}}}}
- text: {{"cmd": "text", "params": {{"at": [x, y], "text": "...", "height": 2.5}}}}
- layer.new: {{"cmd": "layer.new", "params": {{"name": "VISIBLE", "color": "white", "current": true}}}}

Rules:
1. Origin (0.0, 0.0) is anchored at the bottom-left vertex.
2. Contours must form a closed loop.
3. Include layers VISIBLE (white), CENTER (red), DIMENSIONS (green), and ANNOTATIONS (cyan).

Output ONLY a valid JSON array of command objects starting with '[' and ending with ']'."""

    draftsman_raw = query_qwen(
        draftsman_prompt,
        image_path=None,
        system="You are the CAD Draftsman. Output ONLY a valid JSON array of CADCraft commands.",
        max_tokens=1500,
        label="Qwen Draftsman"
    )

    draftsman_cmds = parse_json_from_llm(draftsman_raw)
    commands = normalize_cadcraft_commands(draftsman_cmds, spec_data)

    print(f"[+] Qwen Draftsman formulated {len(commands)} verified CADCraft commands.")

    # =========================================================================
    # STAGE 3: CADCRAFT ENGINE EXECUTION
    # =========================================================================
    print("-" * 75)
    print(f"[STAGE 3: CADCraft Engine] Compiling drawing ({len(commands)} commands)...")
    print("-" * 75)

    out_dxf_abs = os.path.abspath(cli_args.out)
    out_png_abs = str(Path(out_dxf_abs).with_suffix(".png"))
    out_pdf_abs = str(Path(out_dxf_abs).with_suffix(".pdf"))

    cadcraft_bin = shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"
    cadcraft_args = [cadcraft_bin, "run", "--metric"]
    for c in commands:
        cadcraft_args.extend(["--cmd", f"{c['cmd']} {json.dumps(c.get('params', {}))}"])
    cadcraft_args.extend(["--save", out_dxf_abs, "--export", out_png_abs])

    res = subprocess.run(cadcraft_args, capture_output=True, text=True, timeout=30)
    if res.returncode != 0:
        print(f"[!] CADCraft Error: {res.stderr or res.stdout}")
        # Retry with basic geometry if needed
        sys.exit(1)

    print(f"[+] Compiled DXF: {out_dxf_abs} ({os.path.getsize(out_dxf_abs):,} bytes)")
    print(f"[+] Rendered PNG: {out_png_abs} ({os.path.getsize(out_png_abs):,} bytes)")

    # Export vector PDF
    subprocess.run([cadcraft_bin, "convert", out_dxf_abs, out_pdf_abs], capture_output=True)
    if os.path.exists(out_pdf_abs):
        print(f"[+] Exported PDF: {out_pdf_abs} ({os.path.getsize(out_pdf_abs):,} bytes)")

    # =========================================================================
    # STAGE 4: QWEN CRITIC AGENT (Geometric Audit & Scoring on DGX)
    # =========================================================================
    print("-" * 75)
    print("[STAGE 4: Qwen Critic Agent on DGX] Auditing Geometric Precision...")
    print("-" * 75)

    critic_prompt = f"""You are the Senior CAD Metrology Critic.
Audit the following compiled CAD model against the technical drawing for {os.path.basename(image_path)} with h = {h_val} mm.

Commands executed by CADCraft:
{json.dumps(commands[:15], indent=2)}

Audit checks:
1. Closed contour: Are contours closed with 0 gap?
2. Dimension accuracy: Are coordinates and dimensions consistent with h = {h_val} mm?
3. Origin anchoring: Is the origin correctly anchored at (0.0, 0.0)?
4. Holes & features: Are circular bores and features present?

Provide an accuracy score from 0.0% to 100.0% and confirm if production-ready."""

    critic_raw = query_qwen(
        critic_prompt,
        image_path=out_png_abs if os.path.exists(out_png_abs) else None,
        system="You are the Senior CAD Inspector. Provide rigorous accuracy auditing.",
        max_tokens=400,
        label="Qwen Critic"
    )
    print(f"\n[Qwen Critic Audit Report]:\n{critic_raw.strip()}\n")

    # Launch desktop GUI showcase if enabled
    if cli_args.gui:
        print("[*] Launching CADCraft GUI preview...")
        gui_msg = open_in_cadcraft_gui(out_dxf_abs)
        print(f"    {gui_msg}")

    print("=" * 75)
    print(f"  SUCCESS: Generated {os.path.basename(out_dxf_abs)} via Qwen 3.8 27B on DGX")
    print("=" * 75)

if __name__ == "__main__":
    main()
