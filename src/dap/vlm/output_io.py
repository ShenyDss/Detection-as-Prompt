from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from dap.schemas.core import VLMDecision
from dap.utils.jsonl import read_jsonl, write_jsonl
from dap.vlm.output_parser import vlm_decision_from_dict


def read_decisions_jsonl(path: str | Path) -> Iterator[VLMDecision]:
    for row in read_jsonl(path):
        yield vlm_decision_from_dict(row)


def write_decisions_jsonl(path: str | Path, decisions: Iterable[VLMDecision]) -> None:
    write_jsonl(path, (decision.to_dict() for decision in decisions))


def load_decisions(path: str | Path) -> list[VLMDecision]:
    return list(read_decisions_jsonl(path))
