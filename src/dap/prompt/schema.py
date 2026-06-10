from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DetectionPrompt:
    """Structured prompt generated from one detector hypothesis and KG context."""

    image_id: str
    hypothesis_id: str
    image_path: str | None
    region_path: str | None
    route_hint: str | None
    prompt_text: str
    prompt_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectionPrompt":
        return cls(
            image_id=str(data["image_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            image_path=data.get("image_path"),
            region_path=data.get("region_path"),
            route_hint=data.get("route_hint"),
            prompt_text=str(data["prompt_text"]),
            prompt_fields=dict(data.get("prompt_fields", {})),
        )
