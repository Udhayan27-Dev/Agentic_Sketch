import pytest
from sketchcad_agent.agent import DrawingSpec, Entity, Relation


def test_drawing_spec_l_shape():
    """Validates the DrawingSpec model with a hand-written L-shape example.

    L-shape dimensions: 24 mm wide, 12 mm tall, 6 mm wall thickness.
    """
    l_shape_data = {
        "units": "mm",
        "entities": [
            {"id": "e1", "type": "line", "p1": [0.0, 0.0], "p2": [24.0, 0.0]},
            {"id": "e2", "type": "line", "p1": [24.0, 0.0], "p2": [24.0, 6.0]},
            {"id": "e3", "type": "line", "p1": [24.0, 6.0], "p2": [6.0, 6.0]},
            {"id": "e4", "type": "line", "p1": [6.0, 6.0], "p2": [6.0, 12.0]},
            {"id": "e5", "type": "line", "p1": [6.0, 12.0], "p2": [0.0, 12.0]},
            {"id": "e6", "type": "line", "p1": [0.0, 12.0], "p2": [0.0, 0.0]},
        ],
        "relations": [
            {"kind": "coincident", "refs": ["e1", "e2"], "value_mm": None},
            {"kind": "coincident", "refs": ["e2", "e3"], "value_mm": None},
            {"kind": "coincident", "refs": ["e3", "e4"], "value_mm": None},
            {"kind": "coincident", "refs": ["e4", "e5"], "value_mm": None},
            {"kind": "coincident", "refs": ["e5", "e6"], "value_mm": None},
            {"kind": "coincident", "refs": ["e6", "e1"], "value_mm": None},
            {"kind": "horizontal", "refs": ["e1"], "value_mm": None},
            {"kind": "horizontal", "refs": ["e3"], "value_mm": None},
            {"kind": "horizontal", "refs": ["e5"], "value_mm": None},
            {"kind": "vertical", "refs": ["e2"], "value_mm": None},
            {"kind": "vertical", "refs": ["e4"], "value_mm": None},
            {"kind": "vertical", "refs": ["e6"], "value_mm": None},
            {"kind": "distance", "refs": ["e1"], "value_mm": 24.0},
            {"kind": "distance", "refs": ["e6"], "value_mm": 12.0},
            {"kind": "distance", "refs": ["e2"], "value_mm": 6.0},
            {"kind": "distance", "refs": ["e5"], "value_mm": 6.0},
        ],
        "assumptions": ["Origin at bottom-left corner (0,0)"],
        "unreadable": [],
    }

    spec = DrawingSpec.model_validate(l_shape_data)

    assert spec.units == "mm"
    assert len(spec.entities) == 6
    assert len(spec.relations) == 16
    assert spec.entities[0].id == "e1"
    assert spec.entities[0].p1 == [0.0, 0.0]
    assert spec.entities[0].p2 == [24.0, 0.0]
    assert spec.assumptions == ["Origin at bottom-left corner (0,0)"]
    assert spec.unreadable == []

    # Round-trip json serialization test
    json_str = spec.model_dump_json()
    spec_from_json = DrawingSpec.model_validate_json(json_str)
    assert spec_from_json == spec


def test_drawing_spec_invalid_units():
    with pytest.raises(Exception):
        DrawingSpec.model_validate({
            "units": "inches",  # Must be 'mm'
            "entities": [],
            "relations": [],
            "assumptions": [],
            "unreadable": [],
        })


def test_entity_types():
    line = Entity(id="e1", type="line", p1=[0, 0], p2=[10, 0])
    circle = Entity(id="e2", type="circle", center=[0, 0], radius=5.0)
    arc = Entity(id="e3", type="arc", center=[0, 0], radius=5.0, start_deg=0.0, end_deg=90.0)

    assert line.type == "line"
    assert circle.type == "circle"
    assert arc.type == "arc"

    with pytest.raises(Exception):
        Entity(id="e4", type="spline")  # Spline not in Literal['line','circle','arc']


def test_critic_report_schema():
    from sketchcad_agent.agent import CriticReport, Discrepancy

    report = CriticReport(
        accuracy_score=98.5,
        is_accurate=True,
        summary="Drawing matches technical specification accurately.",
        errors=[],
        rectification_commands=[]
    )
    assert report.accuracy_score >= 98.0
    assert report.is_accurate is True

    # Test report with discrepancies
    error_report = CriticReport(
        accuracy_score=85.0,
        is_accurate=False,
        summary="Missing top step line and open loop at e3.",
        errors=[
            Discrepancy(
                issue_type="open_contour",
                description="Vertex between e2 and e3 not coincident",
                how_to_fix="Add coincident constraint between e2.p2 and e3.p1"
            )
        ],
        rectification_commands=["sk.addConstraint(Sketcher.Constraint('Coincident', 1, 2, 2, 1))"]
    )
    assert error_report.accuracy_score < 98.0
    assert len(error_report.errors) == 1
    assert error_report.errors[0].issue_type == "open_contour"

