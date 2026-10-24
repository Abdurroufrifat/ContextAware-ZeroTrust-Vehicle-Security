from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Topology:
    nodes: tuple[str, ...]
    relations: dict[str, tuple[tuple[str, str], ...]]

    @classmethod
    def from_dict(cls, data: dict) -> Topology:
        nodes = tuple(data.get("nodes", ()))
        if not nodes:
            raise ValueError("Topology requires a non-empty 'nodes' list")
        relations: dict[str, tuple[tuple[str, str], ...]] = {}
        for relation, edges in data.get("relations", {}).items():
            parsed: list[tuple[str, str]] = []
            for edge in edges:
                if len(edge) != 2:
                    raise ValueError(f"Invalid edge for relation {relation}: {edge}")
                source, target = str(edge[0]), str(edge[1])
                if source not in nodes or target not in nodes:
                    raise ValueError(f"Unknown topology node in edge {edge}")
                parsed.append((source, target))
            relations[str(relation)] = tuple(parsed)
        if not relations:
            relations["data"] = tuple()
        return cls(nodes=nodes, relations=relations)

    @classmethod
    def from_json(cls, path: str | Path) -> Topology:
        """Load a topology artifact while ignoring provenance metadata."""

        topology_path = Path(path)
        data = json.loads(topology_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(self.relations)

    def relation_adjacency(self, bidirectional: bool = False) -> np.ndarray:
        index = {node: idx for idx, node in enumerate(self.nodes)}
        matrices = np.zeros(
            (len(self.relations), len(self.nodes), len(self.nodes)), dtype=np.float32
        )
        for relation_idx, edges in enumerate(self.relations.values()):
            for source, target in edges:
                matrices[relation_idx, index[target], index[source]] = 1.0
                if bidirectional:
                    matrices[relation_idx, index[source], index[target]] = 1.0
        return matrices

    def validate_feature_order(self, features: Iterable[str]) -> None:
        received = tuple(features)
        if received != self.nodes:
            raise ValueError(
                "Feature order must match topology nodes exactly. "
                f"Expected {self.nodes}; received {received}."
            )
