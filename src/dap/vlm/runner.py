from __future__ import annotations

import json
from typing import Any

from dap.prompt.schema import DetectionPrompt
from dap.schemas.core import VLMDecision
from dap.vlm.mock_reviewer import MockVLMReviewer
from dap.vlm.output_parser import parse_vlm_decision
from dap.vlm.qwen_adapter import QwenVLReviewer


def run_vlm_review(prompts: list[DetectionPrompt], *, vlm_config: dict[str, Any]) -> list[VLMDecision]:
    reviewer = _build_reviewer(vlm_config)
    decisions: list[VLMDecision] = []
    continue_on_error = bool(vlm_config.get("continue_on_error", False))
    for prompt in prompts:
        try:
            raw_output = reviewer.generate(prompt)
            decisions.append(
                parse_vlm_decision(
                    raw_output,
                    image_id=prompt.image_id,
                    hypothesis_id=prompt.hypothesis_id,
                )
            )
        except Exception:
            if not continue_on_error:
                raise
            raw_output = _fallback_revise_output()
            decisions.append(
                parse_vlm_decision(
                    raw_output,
                    image_id=prompt.image_id,
                    hypothesis_id=prompt.hypothesis_id,
                )
            )
    return decisions


def _build_reviewer(vlm_config: dict[str, Any]):
    mode = str(vlm_config.get("mode", "mock")).lower()
    if mode == "mock":
        return MockVLMReviewer()
    if mode in {"qwen_vl", "qwen", "qwen2_vl", "qwen3_vl"}:
        use_lora = bool(vlm_config.get("use_lora", False))
        lora_checkpoint = vlm_config.get("lora_checkpoint") if use_lora else None
        return QwenVLReviewer(
            checkpoint=str(vlm_config.get("checkpoint") or vlm_config.get("model_dir") or ""),
            device=str(vlm_config.get("device", "auto")),
            max_new_tokens=int(vlm_config.get("max_new_tokens", 512)),
            lora_checkpoint=str(lora_checkpoint) if lora_checkpoint else None,
        )
    raise ValueError(f"Unsupported VLM mode: {mode}")


def _fallback_revise_output() -> str:
    return json.dumps(
        {
            "hypothesis_status": "revise",
            "visual_evidence": "The reviewer failed to produce a parseable structured response.",
            "uncertainty_evidence": "Fallback decision emitted by the inference runner.",
            "graph_path": [],
            "corrected_class": None,
            "refined_bbox": None,
            "causes": [],
            "actions": [],
            "risks": ["manual review required"],
        },
        ensure_ascii=False,
    )
