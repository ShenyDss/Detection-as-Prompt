from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class HypothesisStatus(str, Enum):
    VERIFY = "verify"
    REJECT = "reject"
    REVISE = "revise"


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    def to_list(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, values: list[float]) -> "BBox":
        if len(values) != 4:
            raise ValueError(f"BBox expects 4 values, got {len(values)}")
        return cls(*[float(value) for value in values])


@dataclass
class DefectHypothesis:
    """Detector output treated as a structured, reviewable hypothesis."""

    image_id: str
    hypothesis_id: str
    pred_class: str
    bbox: BBox
    score: float
    confusion_candidates: list[str] = field(default_factory=list)
    detector_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = self.bbox.to_list()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefectHypothesis":
        return cls(
            image_id=str(data["image_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            pred_class=str(data["pred_class"]),
            bbox=BBox.from_list(data["bbox"]),
            score=float(data["score"]),
            confusion_candidates=list(data.get("confusion_candidates", [])),
            detector_meta=dict(data.get("detector_meta", {})),
        )


@dataclass
class EvidenceRecord:
    """Evidence fields used to audit preserve/reject/revise decisions."""

    visual: str = ""
    uncertainty: str = ""
    graph_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpertDecision:
    """Graph-grounded decision content for downstream industrial action."""

    causes: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VLMDecision:
    """Structured output expected from the second-stage VLM reviewer."""

    image_id: str
    hypothesis_id: str
    hypothesis_status: HypothesisStatus
    corrected_class: str | None
    refined_bbox: BBox | None
    evidence: EvidenceRecord = field(default_factory=EvidenceRecord)
    decision: ExpertDecision = field(default_factory=ExpertDecision)
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hypothesis_status"] = self.hypothesis_status.value
        data["refined_bbox"] = self.refined_bbox.to_list() if self.refined_bbox else None
        return data
