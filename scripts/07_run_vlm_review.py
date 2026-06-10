from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config
from dap.prompt.io import load_prompts
from dap.vlm.output_io import load_decisions
from dap.vlm.output_io import write_decisions_jsonl
from dap.vlm.output_parser import parse_vlm_decision
from dap.vlm.runner import _build_reviewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM hypothesis review.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--prompts", default="outputs/predictions/detection_prompts.jsonl")
    parser.add_argument("--output", default="outputs/predictions/vlm_decisions.jsonl")
    parser.add_argument("--mode", default=None, choices=["mock", "qwen_vl", "qwen", "qwen2_vl", "qwen3_vl"])
    parser.add_argument("--limit", type=int, default=None, help="Review only the first N prompts.")
    parser.add_argument("--use-lora", action="store_true", help="Load the LoRA adapter from config or --lora-checkpoint.")
    parser.add_argument("--lora-checkpoint", default=None, help="Path to a PEFT LoRA adapter directory.")
    parser.add_argument("--continue-on-error", action="store_true", help="Write fallback revise decisions if a sample fails.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output and skip completed hypothesis ids.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    vlm_config = dict(config.vlm)
    if args.mode:
        vlm_config["mode"] = args.mode
    if args.use_lora:
        vlm_config["use_lora"] = True
    if args.lora_checkpoint:
        vlm_config["use_lora"] = True
        vlm_config["lora_checkpoint"] = args.lora_checkpoint
    if args.continue_on_error:
        vlm_config["continue_on_error"] = True

    prompts = load_prompts(PROJECT_ROOT / args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    output_path = PROJECT_ROOT / args.output
    completed_ids = set()
    decisions = []
    if args.resume and output_path.exists():
        decisions = load_decisions(output_path)
        completed_ids = {decision.hypothesis_id for decision in decisions}
        prompts = [prompt for prompt in prompts if prompt.hypothesis_id not in completed_ids]

    if not args.resume:
        write_decisions_jsonl(output_path, [])

    reviewer = _build_reviewer(vlm_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as file:
        for index, prompt in enumerate(prompts, start=1):
            try:
                raw_output = reviewer.generate(prompt)
                decision = parse_vlm_decision(
                    raw_output,
                    image_id=prompt.image_id,
                    hypothesis_id=prompt.hypothesis_id,
                )
            except Exception:
                if not args.continue_on_error:
                    raise
                raw_output = _fallback_revise_output()
                decision = parse_vlm_decision(
                    raw_output,
                    image_id=prompt.image_id,
                    hypothesis_id=prompt.hypothesis_id,
                )
            decisions.append(decision)
            file.write(json.dumps(decision.to_dict(), ensure_ascii=False, separators=(",", ":")))
            file.write("\n")
            file.flush()
            if index == 1 or index % 10 == 0 or index == len(prompts):
                print(f"Reviewed {index}/{len(prompts)} new prompts")

    counts = Counter(decision.hypothesis_status.value for decision in decisions)
    print(f"Wrote {len(decisions)} VLM decisions to {output_path}")
    print("Decision summary:")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")


def _fallback_revise_output() -> str:
    return json.dumps(
        {
            "hypothesis_status": "revise",
            "visual_evidence": "The reviewer failed to produce a parseable structured response.",
            "uncertainty_evidence": "Fallback decision emitted by the inference script.",
            "graph_path": [],
            "corrected_class": None,
            "refined_bbox": None,
            "causes": [],
            "actions": [],
            "risks": ["manual review required"],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
