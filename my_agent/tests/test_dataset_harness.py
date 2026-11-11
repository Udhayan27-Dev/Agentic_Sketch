"""Unit and integration tests for the CADCraft Dataset Generator and Evaluation Harness."""

import json
import os
import shutil
import tempfile
import pytest

from harness.dataset_generator import (
    ShapeGenerator,
    generate_sample_by_index,
    render_drawing_image,
    execute_sample_in_cadcraft,
)
from harness.evaluator import CADGeometryEvaluator
from training.prepare_vlm_data import convert_sample, format_assistant_response


def test_shape_generator_all_five_families():
    """Validates generation of all 5 mechanical shape families."""
    for i in range(5):
        sample = generate_sample_by_index(i, split="test")
        assert sample is not None
        assert sample.sample_id == f"test_{i:04d}"
        assert len(sample.entities) >= 4
        assert len(sample.cadcraft_commands) >= 2
        assert len(sample.cadcraft_script) > 10
        assert sample.bounds[2] > sample.bounds[0]
        assert sample.bounds[3] > sample.bounds[1]


def test_drawing_renderer():
    """Validates Pillow technical drawing rendering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample = ShapeGenerator.generate_corner_fillet_plate("render_test", seed=42)
        out_png = os.path.join(tmpdir, "test.png")
        render_drawing_image(sample, out_png)
        assert os.path.exists(out_png)
        assert os.path.getsize(out_png) > 1000


def test_cadcraft_execution():
    """Validates that cadcraft-cli executes sample commands and produces valid DXF."""
    cadcraft_cli = shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"
    if not os.path.exists(cadcraft_cli):
        pytest.skip("cadcraft-cli not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        sample = ShapeGenerator.generate_tangent_pulley("pulley_test", seed=10)
        ok, dxf_path, msg = execute_sample_in_cadcraft(sample, tmpdir)
        assert ok is True, f"CADCraft execution failed: {msg}"
        assert os.path.exists(dxf_path)
        assert os.path.getsize(dxf_path) > 500


def test_evaluator_on_perfect_shape():
    """Validates that the geometric evaluator scores ground-truth shapes at 100%."""
    cadcraft_cli = shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"
    if not os.path.exists(cadcraft_cli):
        pytest.skip("cadcraft-cli not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        sample = ShapeGenerator.generate_corner_fillet_plate("eval_test", seed=123)
        ok, dxf_path, _ = execute_sample_in_cadcraft(sample, tmpdir)
        assert ok is True

        # Build spec dict
        from harness.generate_dataset import sample_to_drawing_spec_dict
        gt_spec = sample_to_drawing_spec_dict(sample)

        evaluator = CADGeometryEvaluator()
        report = evaluator.evaluate_dxf(dxf_path, gt_spec, sample_id="eval_test")

        assert report.cadcraft_success is True
        assert report.is_watertight is True
        assert report.max_gap_mm <= 0.05
        assert report.tangency_mean_error_deg <= 0.1
        assert report.accuracy_score >= 95.0


def test_vlm_data_formatter():
    """Validates VLM training data formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sample = ShapeGenerator.generate_slotted_link("vlm_test", seed=7)
        img_path = os.path.join(tmpdir, "vlm_test_drawing.png")
        render_drawing_image(sample, img_path)

        from harness.generate_dataset import sample_to_drawing_spec_dict
        spec_dict = sample_to_drawing_spec_dict(sample)
        spec_path = os.path.join(tmpdir, "vlm_test_spec.json")
        with open(spec_path, "w") as f:
            json.dump(spec_dict, f)

        cmds_path = os.path.join(tmpdir, "vlm_test_cmds.json")
        with open(cmds_path, "w") as f:
            json.dump(sample.cadcraft_commands, f)

        scr_path = os.path.join(tmpdir, "vlm_test.scr")
        with open(scr_path, "w") as f:
            f.write(sample.cadcraft_script)

        sample_entry = {
            "sample_id": "vlm_test",
            "category": sample.category,
            "description": sample.description,
            "drawing_image": "vlm_test_drawing.png",
            "drawing_spec": "vlm_test_spec.json",
            "cadcraft_commands": "vlm_test_cmds.json",
            "cadcraft_script": "vlm_test.scr",
        }

        openai_row = convert_sample(sample_entry, tmpdir, format_type="openai")
        assert "messages" in openai_row
        assert len(openai_row["messages"]) == 3

        qwen_row = convert_sample(sample_entry, tmpdir, format_type="qwen")
        assert "conversations" in qwen_row
        assert len(qwen_row["conversations"]) == 3


def test_plug_gauge_image_to_cad_harness():
    """Validates the image-to-CAD harness on plug_gauge_harness files."""
    cadcraft_cli = shutil.which("cadcraft-cli") or "/usr/bin/cadcraft-cli"
    if not os.path.exists(cadcraft_cli):
        pytest.skip("cadcraft-cli not installed")

    from harness.image_to_cad_harness import ImageCADHarness
    img_path = "harness/plug_gauge_harness.png"
    if not os.path.exists(img_path):
        pytest.skip(f"Test drawing {img_path} not found")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dxf = os.path.join(tmpdir, "plug_gauge_test.dxf")
        harness = ImageCADHarness(cadcraft_cli=cadcraft_cli)
        result = harness.process_image(img_path, out_dxf, open_gui=False)
        assert result.success is True
        assert result.evaluation.accuracy_score >= 98.0
        assert result.evaluation.is_watertight is True
        assert os.path.exists(out_dxf)
