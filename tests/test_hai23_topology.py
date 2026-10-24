import json
from pathlib import Path

import numpy as np

from coact_cps.data.topology import Topology


def test_external_topology_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "topology.json"
    path.write_text(
        json.dumps(
            {
                "nodes": ["a", "b"],
                "relations": {"physical": [["a", "b"]]},
                "provenance": {"physical": {"source": "test"}},
            }
        ),
        encoding="utf-8",
    )
    topology = Topology.from_json(path)
    topology.validate_feature_order(["a", "b"])
    adjacency = topology.relation_adjacency(bidirectional=True)
    assert adjacency.shape == (1, 2, 2)
    assert np.array_equal(adjacency[0], np.array([[0, 1], [1, 0]], dtype=np.float32))
