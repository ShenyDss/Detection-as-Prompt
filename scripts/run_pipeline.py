from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".conda" / "dap" / "python.exe"


PIPELINE_STEPS = [
    ["scripts/01_run_detector.py", "--mode", "mock"],
    ["scripts/06_route_hypotheses.py"],
    ["scripts/05_build_prompts.py"],
    ["scripts/07_run_vlm_review.py", "--mode", "mock"],
    ["scripts/08_write_demo_ground_truth.py"],
    ["scripts/08_evaluate.py"],
    ["scripts/09_run_ablation.py"],
    ["scripts/10_build_instruction_data.py"],
    ["scripts/10_train_lora.py", "--dry-run"],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Detection-as-Prompt pipeline.")
    parser.add_argument("--skip-demo-gt", action="store_true", help="Do not overwrite demo ground truth.")
    parser.add_argument("--stop-after", default=None, help="Stop after a script basename, e.g. 05_build_prompts.py.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for step in PIPELINE_STEPS:
        if args.skip_demo_gt and step[0].endswith("08_write_demo_ground_truth.py"):
            continue
        print(f"\n[Pipeline] Running: {' '.join(step)}")
        subprocess.run([str(PYTHON), *step], cwd=PROJECT_ROOT, check=True)
        if args.stop_after and step[0].endswith(args.stop_after):
            break


if __name__ == "__main__":
    main()
