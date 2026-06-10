from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from dap.schemas.core import DefectHypothesis
from dap.utils.jsonl import read_jsonl, write_jsonl


def read_hypotheses_jsonl(path: str | Path) -> Iterator[DefectHypothesis]:
    for row in read_jsonl(path):
        yield DefectHypothesis.from_dict(row)


def write_hypotheses_jsonl(path: str | Path, hypotheses: Iterable[DefectHypothesis]) -> None:
    write_jsonl(path, (hypothesis.to_dict() for hypothesis in hypotheses))


def load_hypotheses(path: str | Path) -> list[DefectHypothesis]:
    return list(read_hypotheses_jsonl(path))
