from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - used only when PyYAML is absent.
    yaml = None


@dataclass(frozen=True)
class AppConfig:
    """Container for the five YAML files used by the project."""

    dataset: dict[str, Any]
    detector: dict[str, Any]
    vlm: dict[str, Any]
    kg: dict[str, Any]
    experiment: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "detector": self.detector,
            "vlm": self.vlm,
            "kg": self.kg,
            "experiment": self.experiment,
        }


def read_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        text = file.read()

    data = yaml.safe_load(text) if yaml is not None else _parse_simple_yaml(text)
    data = data or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")

    return data


def load_app_config(config_dir: str | Path = "configs") -> AppConfig:
    root = Path(config_dir)
    return AppConfig(
        dataset=read_yaml(root / "dataset.yaml"),
        detector=read_yaml(root / "detector.yaml"),
        vlm=read_yaml(root / "vlm.yaml"),
        kg=read_yaml(root / "kg.yaml"),
        experiment=read_yaml(root / "experiment.yaml"),
    )


def _parse_simple_yaml(text: str) -> Any:
    """Small fallback parser for the simple YAML files in this scaffold.

    It supports nested mappings, list items, strings, numbers, booleans, and null.
    Install PyYAML for full YAML support once the runtime environment is ready.
    """

    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))

    def parse_scalar(value: str) -> Any:
        if value in {"null", "None", "~"}:
            return None
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index

        current_indent, current_text = lines[index]
        if current_indent < indent:
            return {}, index

        if current_text.startswith("- "):
            values: list[Any] = []
            while index < len(lines):
                line_indent, line_text = lines[index]
                if line_indent != indent or not line_text.startswith("- "):
                    break
                item_text = line_text[2:].strip()
                index += 1
                if item_text:
                    values.append(parse_scalar(item_text))
                else:
                    child, index = parse_block(index, indent + 2)
                    values.append(child)
            return values, index

        values: dict[str, Any] = {}
        while index < len(lines):
            line_indent, line_text = lines[index]
            if line_indent != indent or line_text.startswith("- "):
                break
            key, separator, value_text = line_text.partition(":")
            if not separator:
                raise ValueError(f"Invalid YAML line: {line_text}")
            index += 1
            value_text = value_text.strip()
            if value_text:
                values[key.strip()] = parse_scalar(value_text)
            else:
                child, index = parse_block(index, indent + 2)
                values[key.strip()] = child
        return values, index

    parsed, next_index = parse_block(0, lines[0][0] if lines else 0)
    if next_index != len(lines):
        raise ValueError("Could not parse the complete YAML document.")
    return parsed
