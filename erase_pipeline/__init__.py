"""
Dimension-Erase Pipeline (Mask-only, NO redrawing)
"""

from erase_pipeline.pipeline import DimensionErasePipeline
from erase_pipeline.schemas import (
    DimensionAnnotation,
    DimensionsFile,
    SegmentFeatures,
    ClassificationResult,
    VerificationReport
)

__all__ = [
    "DimensionErasePipeline",
    "DimensionAnnotation",
    "DimensionsFile",
    "SegmentFeatures",
    "ClassificationResult",
    "VerificationReport"
]
