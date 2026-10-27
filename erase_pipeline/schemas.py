"""
Data schemas for the Dimension-Erase Pipeline.
JSON contracts for dimensions, segments, classification, and verification.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class DimensionAnnotation(BaseModel):
    """Extracted engineering dimension annotation."""
    raw: str = Field(..., description="Raw text string as detected, e.g. 'R 0,55h'")
    type: Literal["radius", "linear", "angle", "label", "symbol"] = Field(
        ..., description="Type of dimension callout"
    )
    value: Optional[float] = Field(None, description="Normalized numeric value, e.g. 0.55")
    unit: Optional[str] = Field(None, description="Normalized unit, e.g. 'h', 'mm', '°'")
    bbox: List[int] = Field(..., description="Bounding box [x1, y1, x2, y2] in normalized coordinates")


class DimensionsFile(BaseModel):
    """Root contract for dimensions.json."""
    image_shape: List[int]
    total_dimensions: int
    dimensions: List[DimensionAnnotation]


class SegmentFeatures(BaseModel):
    """Features computed for each segmented stroke edge / component."""
    id: int
    area: int
    bbox: List[int]  # [x, y, w, h]
    stroke_width: float
    mean_gray: float
    fill_ratio: float
    is_on_closed_loop: bool
    touches_text_box: bool
    free_endpoints: int


class ClassificationResult(BaseModel):
    """Segment classification outcome."""
    erase_ids: List[int] = Field(default_factory=list)
    keep_ids: List[int] = Field(default_factory=list)
    unsure_ids: List[int] = Field(default_factory=list)
    vlm_classified_ids: List[int] = Field(default_factory=list)
    rule_reasons: Dict[int, str] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    """Root contract for report.json."""
    session_id: str
    total_segments: int
    erased_segments: int
    kept_segments: int
    leftover_components: int
    cleanliness_score: float
    iou_score: Optional[float] = None
    chamfer_distance: Optional[float] = None
    vlm_consulted_segments: List[int] = Field(default_factory=list)
    dimensions_erased_count: int = 0
    watertight: bool = True
