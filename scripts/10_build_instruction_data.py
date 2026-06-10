from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.evaluation.ground_truth import load_ground_truth
from dap.prompt.io import load_prompts
from dap.schemas.hypothesis_io import load_hypotheses
from dap.training import build_instruction_samples
from dap.training.io import write_instruction_jsonl
from dap.vlm.output_io import load_decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build correction-oriented VLM instruction data.")
    parser.add_argument("--prompts", default="outputs/predictions/detection_prompts.jsonl")
    parser.add_argument("--hypotheses", default="data/detector_predictions/hypotheses.jsonl")
    parser.add_argument("--decisions", default=None)
    parser.add_argument("--ground-truth", default="data/annotations/test_decisions.jsonl")
    parser.add_argument("--output", default="data/instruction_data/dap_instruction_train.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = load_prompts(PROJECT_ROOT / args.prompts)
    hypotheses = load_hypotheses(PROJECT_ROOT / args.hypotheses)
    decisions = load_decisions(PROJECT_ROOT / args.decisions) if args.decisions else []
    ground_truth = load_ground_truth(PROJECT_ROOT / args.ground_truth)

    samples = build_instruction_samples(
        prompts=prompts,
        hypotheses=hypotheses,
        decisions=decisions,
        ground_truth=ground_truth,
    )
    output_path = PROJECT_ROOT / args.output
    write_instruction_jsonl(output_path, samples)
    print(f"Wrote {len(samples)} instruction samples to {output_path}")
    if samples:
        print("First sample target:")
        print(samples[0].target)


if __name__ == "__main__":
    main()
