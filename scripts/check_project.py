from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    checks = {
        "python": sys.version.split()[0],
        "torch": importlib.util.find_spec("torch") is not None,
        "ultralytics": importlib.util.find_spec("ultralytics") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "peft": importlib.util.find_spec("peft") is not None,
        "datasets": importlib.util.find_spec("datasets") is not None,
        "accelerate": importlib.util.find_spec("accelerate") is not None,
        "modelscope": importlib.util.find_spec("modelscope") is not None,
        "swift": importlib.util.find_spec("swift") is not None,
        "configs_present": all((PROJECT_ROOT / "configs" / name).exists() for name in [
            "dataset.yaml", "detector.yaml", "vlm.yaml", "kg.yaml", "experiment.yaml",
            "training.yaml", "baseline.yaml", "ablation.yaml", "modelscope.yaml",
        ]),
    }
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
