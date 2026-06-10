from __future__ import annotations

import ast
import json
import re
from typing import Any

from dap.schemas.core import BBox, EvidenceRecord, ExpertDecision, HypothesisStatus, VLMDecision


def parse_vlm_decision(
    raw_output: str,
    *,
    image_id: str,
    hypothesis_id: str,
) -> VLMDecision:
    data = _parse_json_like(raw_output)
    data.setdefault("image_id", image_id)
    data.setdefault("hypothesis_id", hypothesis_id)
    data["raw_output"] = raw_output
    return vlm_decision_from_dict(data)


def vlm_decision_from_dict(data: dict[str, Any]) -> VLMDecision:
    status = _parse_status(data.get("hypothesis_status", data.get("status", "revise")))
    corrected_class = data.get("corrected_class")
    if corrected_class in {"", "null", "None"}:
        corrected_class = None

    refined_bbox = data.get("refined_bbox")
    bbox = BBox.from_list(refined_bbox) if isinstance(refined_bbox, list) else None

    nested_evidence = dict(data.get("evidence", {})) if isinstance(data.get("evidence"), dict) else {}
    nested_decision = dict(data.get("decision", {})) if isinstance(data.get("decision"), dict) else {}

    graph_path = data.get("graph_path", nested_evidence.get("graph_path", []))
    if not isinstance(graph_path, list):
        graph_path = [str(graph_path)]

    evidence = EvidenceRecord(
        visual=str(data.get("visual_evidence", nested_evidence.get("visual", ""))),
        uncertainty=str(data.get("uncertainty_evidence", nested_evidence.get("uncertainty", ""))),
        graph_path=[str(item) for item in graph_path],
    )
    decision = ExpertDecision(
        causes=_as_str_list(data.get("causes", nested_decision.get("causes", []))),
        actions=_as_str_list(data.get("actions", nested_decision.get("actions", []))),
        risks=_as_str_list(data.get("risks", nested_decision.get("risks", []))),
    )
    return VLMDecision(
        image_id=str(data["image_id"]),
        hypothesis_id=str(data["hypothesis_id"]),
        hypothesis_status=status,
        corrected_class=str(corrected_class) if corrected_class is not None else None,
        refined_bbox=bbox,
        evidence=evidence,
        decision=decision,
        raw_output=data.get("raw_output"),
    )


def _parse_status(value: Any) -> HypothesisStatus:
    text = str(value).strip().lower()
    if text not in {item.value for item in HypothesisStatus}:
        raise ValueError(f"Invalid hypothesis_status: {value}")
    return HypothesisStatus(text)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _parse_json_like(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    text = _extract_json_object(text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _parse_python_literal(text)
    if not isinstance(value, dict):
        raise ValueError("VLM output must parse to a JSON object.")
    return value


def _extract_json_object(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _parse_python_literal(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        repaired = _repair_common_json_issues(text)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as json_exc:
            raise ValueError(f"Could not parse VLM output as JSON-like object: {text[:200]}") from json_exc


def _repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = repaired.replace("None", "null").replace("True", "true").replace("False", "false")
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)
    return repaired
