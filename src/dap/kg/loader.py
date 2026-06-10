from __future__ import annotations

import json
from pathlib import Path

from dap.kg.schema import DefectKnowledgeGraph


def load_defect_kg(path: str | Path) -> DefectKnowledgeGraph:
    kg_path = Path(path)
    if not kg_path.exists():
        raise FileNotFoundError(f"Knowledge graph file not found: {kg_path}")

    with kg_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Knowledge graph must be a JSON object: {kg_path}")
    return DefectKnowledgeGraph.from_dict(data)


def save_defect_kg(path: str | Path, graph: DefectKnowledgeGraph) -> None:
    kg_path = Path(path)
    kg_path.parent.mkdir(parents=True, exist_ok=True)
    with kg_path.open("w", encoding="utf-8") as file:
        json.dump(graph.to_dict(), file, ensure_ascii=False, indent=2)
