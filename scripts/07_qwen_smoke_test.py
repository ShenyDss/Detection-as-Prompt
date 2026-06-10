from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.prompt.io import load_prompts
from dap.vlm.output_parser import parse_vlm_decision
from dap.vlm.qwen_adapter import QwenVLReviewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a one-sample Qwen-VL smoke test.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--prompts", default="outputs/predictions/detection_prompts.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--use-lora", action="store_true", help="Load the LoRA adapter from config or --lora-checkpoint.")
    parser.add_argument("--lora-checkpoint", default=None, help="Path to a PEFT LoRA adapter directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    prompts = load_prompts(PROJECT_ROOT / args.prompts)
    if not prompts:
        raise ValueError(f"No prompts found at {args.prompts}")
    prompt = prompts[args.index]

    vlm_config = dict(config.vlm)
    checkpoint = str(vlm_config.get("checkpoint") or vlm_config.get("model_dir") or "")
    lora_checkpoint = None
    if args.use_lora:
        lora_checkpoint = str(args.lora_checkpoint or vlm_config.get("lora_checkpoint") or "")
    reviewer = QwenVLReviewer(
        checkpoint=checkpoint,
        device=str(vlm_config.get("device", "auto")),
        max_new_tokens=args.max_new_tokens,
        lora_checkpoint=lora_checkpoint,
    )
    raw_output = reviewer.generate(prompt)
    decision = parse_vlm_decision(
        raw_output,
        image_id=prompt.image_id,
        hypothesis_id=prompt.hypothesis_id,
    )

    print("Raw output:")
    print(raw_output)
    print("")
    print("Parsed decision:")
    print(decision.to_dict())


if __name__ == "__main__":
    main()
