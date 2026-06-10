from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from dap.routing.schema import RouteDecision
from dap.utils.jsonl import read_jsonl, write_jsonl


def read_routes_jsonl(path: str | Path) -> Iterator[RouteDecision]:
    for row in read_jsonl(path):
        yield RouteDecision.from_dict(row)


def write_routes_jsonl(path: str | Path, routes: Iterable[RouteDecision]) -> None:
    write_jsonl(path, (route.to_dict() for route in routes))


def load_routes(path: str | Path) -> list[RouteDecision]:
    return list(read_routes_jsonl(path))


def load_route_lookup(path: str | Path) -> dict[str, RouteDecision]:
    return {route.hypothesis_id: route for route in read_routes_jsonl(path)}
