"""Headless CADCraft and vector rendering utilities for production SketchCAD.

Decouples CAD generation and inspection from desktop GUI (X11) requirements.
Provides SVG vector export for browser canvas rendering and PNG rendering
for the visual Critic audit loop.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt


def render_dxf_to_svg(dxf_path: str, svg_output_path: str) -> bool:
    """Converts a watertight DXF file to a clean, scalable vector SVG for browser display.

    Args:
        dxf_path: Path to the input .dxf file.
        svg_output_path: Target path for the generated .svg file.

    Returns:
        bool: True if conversion succeeded, False otherwise.
    """
    if not os.path.exists(dxf_path):
        return False

    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        fig = plt.figure(figsize=(10, 10), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)

        ax.axis("off")
        os.makedirs(os.path.dirname(os.path.abspath(svg_output_path)), exist_ok=True)
        fig.savefig(svg_output_path, format="svg", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[render_dxf_to_svg] Error rendering {dxf_path} to SVG: {e}")
        return False


def render_dxf_to_png(dxf_path: str, png_output_path: str, dpi: int = 300) -> bool:
    """Renders a high-resolution raster image of the DXF for Critic visual audit.

    Args:
        dxf_path: Path to the input .dxf file.
        png_output_path: Target path for the output .png file.
        dpi: Resolution density (defaults to 300 DPI).

    Returns:
        bool: True if render succeeded, False otherwise.
    """
    if not os.path.exists(dxf_path):
        return False

    try:
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        fig = plt.figure(figsize=(8, 8), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)

        ax.axis("off")
        os.makedirs(os.path.dirname(os.path.abspath(png_output_path)), exist_ok=True)
        fig.savefig(png_output_path, format="png", bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        return True
    except Exception as e:
        print(f"[render_dxf_to_png] Error rendering {dxf_path} to PNG: {e}")
        return False


def run_cadcraft_headless(
    script_path: str,
    out_dxf_path: str,
    workspace_dir: Optional[str] = None
) -> Tuple[bool, str]:
    """Executes CADCraft CLI non-interactively in headless mode.

    Args:
        script_path: Path to the CADCraft DSL / script.
        out_dxf_path: Path where the output DXF must be written.
        workspace_dir: Working directory for execution.

    Returns:
        Tuple[bool, str]: (Success, error_or_output_message)
    """
    cadcraft_bin = shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"
    if not os.path.exists(cadcraft_bin):
        return False, f"cadcraft-cli binary not found at {cadcraft_bin}"

    cmd = [cadcraft_bin, "run", "--input", script_path, "--output", out_dxf_path]
    cwd = workspace_dir or os.path.dirname(os.path.abspath(out_dxf_path))

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60
        )
        if result.returncode == 0:
            return True, result.stdout
        return False, f"CADCraft error (code {result.returncode}): {result.stderr}"
    except Exception as e:
        return False, f"Exception executing CADCraft CLI: {e}"
