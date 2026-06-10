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
    parser = argparse.ArgumentParser(description="Check local Qwen-VL model files.")
    parser.add_argument("--config", default="configs/modelscope.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(PROJECT_ROOT / args.config)
    model_dir = PROJECT_ROOT / config["local_dir"]
    print(f"Model dir: {model_dir}")
    print(f"Exists: {model_dir.exists()}")

    weight_files = sorted(model_dir.glob("*.safetensors")) + sorted(model_dir.glob("*.bin"))
    temp_weights = sorted((model_dir / "._____temp").glob("*")) if (model_dir / "._____temp").exists() else []
    print(f"Root weight files: {[path.name for path in weight_files]}")
    print(f"Temp files: {[path.name for path in temp_weights]}")

    try:
        from transformers import AutoConfig, AutoProcessor
        config_obj = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        print(f"Transformers config: {type(config_obj).__name__}, model_type={getattr(config_obj, 'model_type', None)}")
        print(f"Processor: {type(processor).__name__}")
    except Exception as exc:
        print(f"Config/processor check failed: {exc}")

    for temp_file in temp_weights:
        if temp_file.suffix == ".safetensors":
            try:
                from safetensors import safe_open
                with safe_open(temp_file, framework="pt", device="cpu") as file:
                    print(f"Temp safetensors readable: {temp_file.name}, tensors={len(file.keys())}")
            except Exception as exc:
                print(f"Temp safetensors incomplete or unreadable: {temp_file.name}: {exc}")

    if not weight_files:
        print("Status: incomplete. Re-run scripts/download_modelscope_model.py until root weight files appear.")
    else:
        print("Status: root weights found.")


if __name__ == "__main__":
    main()
