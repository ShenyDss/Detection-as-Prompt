from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DefectClassNode:
    """Domain knowledge attached to one defect category."""

    name: str
    visual_attributes: list[str] = field(default_factory=list)
    confusion_classes: list[str] = field(default_factory=list)
    causes: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefectClassNode":
        return cls(
            name=str(data["name"]),
            visual_attributes=list(data.get("visual_attributes", [])),
            confusion_classes=list(data.get("confusion_classes", [])),
            causes=list(data.get("causes", [])),
            actions=list(data.get("actions", [])),
            risk_notes=list(data.get("risk_notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DefectKnowledgeGraph:
    """Compact JSON-backed defect knowledge graph."""

    name: str
    classes: dict[str, DefectClassNode]
    cause_action_edges: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DefectKnowledgeGraph":
        class_items = data.get("classes", [])
        if not isinstance(class_items, list):
            raise ValueError("KG field 'classes' must be a list.")

        classes = {node.name: node for node in (DefectClassNode.from_dict(item) for item in class_items)}
        edges = {
            str(cause): [str(action) for action in actions]
            for cause, actions in dict(data.get("cause_action_edges", {})).items()
        }
        return cls(
            name=str(data.get("name", "defect_knowledge_graph")),
            classes=classes,
            cause_action_edges=edges,
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "classes": [node.to_dict() for node in self.classes.values()],
            "cause_action_edges": self.cause_action_edges,
            "metadata": self.metadata,
        }

    def has_class(self, class_name: str) -> bool:
        return class_name in self.classes

    def get_class(self, class_name: str) -> DefectClassNode:
        try:
            return self.classes[class_name]
        except KeyError as exc:
            raise KeyError(f"Unknown defect class in KG: {class_name}") from exc


@dataclass
class KGContext:
    """Category-specific graph context used by the prompt constructor."""

    class_name: str
    visual_attributes: list[str]
    confusion_classes: list[str]
    causes: list[str]
    actions: list[str]
    risk_notes: list[str]
    cause_action_paths: list[list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
