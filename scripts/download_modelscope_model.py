from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config.loader import read_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model from ModelScope.")
    parser.add_argument("--config", default="configs/modelscope.yaml")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--local-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(PROJECT_ROOT / args.config)
    model_id = args.model_id or config["model_id"]
    local_dir = PROJECT_ROOT / (args.local_dir or config["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "modelscope is not installed. Install it with: pip install modelscope"
        ) from exc

    path = snapshot_download(
        model_id,
        local_dir=str(local_dir),
        revision=config.get("revision", "master"),
        ignore_file_pattern=config.get("ignore_file_pattern"),
        max_workers=int(config.get("max_workers", 4)),
    )
    print(f"Downloaded {model_id} to {path}")


if __name__ == "__main__":
    main()
