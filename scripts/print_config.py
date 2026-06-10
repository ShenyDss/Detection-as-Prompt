from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dap.config import load_app_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print Detection-as-Prompt config.")
    parser.add_argument("--config-dir", default="configs", help="Directory containing YAML config files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(PROJECT_ROOT / args.config_dir)
    print(json.dumps(config.as_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
