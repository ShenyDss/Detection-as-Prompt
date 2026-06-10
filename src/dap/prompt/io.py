from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from dap.prompt.schema import DetectionPrompt
from dap.utils.jsonl import read_jsonl, write_jsonl


def read_prompts_jsonl(path: str | Path) -> Iterator[DetectionPrompt]:
    for row in read_jsonl(path):
        yield DetectionPrompt.from_dict(row)


def write_prompts_jsonl(path: str | Path, prompts: Iterable[DetectionPrompt]) -> None:
    write_jsonl(path, (prompt.to_dict() for prompt in prompts))


def load_prompts(path: str | Path) -> list[DetectionPrompt]:
    return list(read_prompts_jsonl(path))
