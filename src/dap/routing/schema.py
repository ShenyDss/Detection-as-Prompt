from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ReviewMode(str, Enum):
    LIGHT_VERIFICATION = "light_verification"
    CLASS_COMPARISON = "class_comparison"
    FALSE_ALARM_CHECKING = "false_alarm_checking"
    BOX_REFINEMENT = "box_refinement"


@dataclass
class UncertaintySignals:
    confidence: float
    class_margin: float
    bbox_quality: float
    top_class_score: float | None = None
    second_class_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RouteDecision:
    image_id: str
    hypothesis_id: str
    review_mode: ReviewMode
    uncertainty: UncertaintySignals
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["review_mode"] = self.review_mode.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteDecision":
        return cls(
            image_id=str(data["image_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            review_mode=ReviewMode(str(data["review_mode"])),
            uncertainty=UncertaintySignals(**dict(data["uncertainty"])),
            reasons=list(data.get("reasons", [])),
        )
