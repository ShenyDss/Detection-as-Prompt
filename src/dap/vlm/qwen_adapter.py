from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dap.prompt.schema import DetectionPrompt


class QwenVLReviewer:
    """Thin adapter for local Qwen-VL/Qwen3-VL inference."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 512,
        lora_checkpoint: str | None = None,
    ):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "transformers is not installed. Install transformers, accelerate, and the target Qwen-VL "
                "dependencies before using vlm.mode='qwen_vl'."
            ) from exc

        self.checkpoint = checkpoint
        self.lora_checkpoint = lora_checkpoint
        self.device = device
        self.max_new_tokens = max_new_tokens
        if not checkpoint:
            raise ValueError("A local Qwen-VL checkpoint path is required.")
        checkpoint = str(_resolve_existing_path(checkpoint) or checkpoint)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            dtype="auto",
            device_map=device,
        )
        if lora_checkpoint:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "peft is not installed. Install peft before loading a LoRA adapter."
                ) from exc
            resolved_lora = str(_resolve_existing_path(lora_checkpoint) or lora_checkpoint)
            self.model = PeftModel.from_pretrained(self.model, resolved_lora)
        self.model.eval()

    def generate(self, prompt: DetectionPrompt) -> str:
        image_path = _select_image_path(prompt)
        messages = [
            {
                "role": "user",
                "content": [],
            }
        ]
        if image_path is not None:
            messages[0]["content"].append({"type": "image", "image": str(image_path)})
        messages[0]["content"].append({"type": "text", "text": prompt.prompt_text})

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(_model_device(self.model))
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return _ensure_json_like(decoded)


def _select_image_path(prompt: DetectionPrompt) -> Path | None:
    for raw_path in (prompt.region_path, prompt.image_path):
        path = _resolve_existing_path(raw_path)
        if path is not None:
            return path
    return None


def _resolve_existing_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists():
            return resolved
    return None


def _model_device(model: Any):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _ensure_json_like(text: str) -> str:
    if "{" in text and "}" in text:
        return text
    return json.dumps(
        {
            "hypothesis_status": "revise",
            "visual_evidence": text,
            "uncertainty_evidence": "",
            "graph_path": [],
            "corrected_class": None,
            "refined_bbox": None,
            "causes": [],
            "actions": [],
            "risks": [],
        },
        ensure_ascii=False,
    )
