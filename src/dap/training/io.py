from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from dap.training.schema import InstructionSample
from dap.utils.jsonl import read_jsonl, write_jsonl


def write_instruction_jsonl(path: str | Path, samples: Iterable[InstructionSample]) -> None:
    write_jsonl(path, (sample.to_dict() for sample in samples))


def read_instruction_jsonl(path: str | Path) -> Iterator[dict]:
    yield from read_jsonl(path)
