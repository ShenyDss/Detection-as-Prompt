from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class InstructionSample:
    """Correction-oriented VLM instruction-tuning sample."""

    sample_id: str
    image_id: str
    hypothesis_id: str
    image_path: str | None
    region_path: str | None
    prompt: str
    target: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
