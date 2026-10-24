from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

# Projected from the official HAI boiler physical graph. The integer denotes
# the graph's published dynamics code: 1=fluid, 2=measurement, 3=thermal.
# Device nodes without an observed HAI-23.05 telemetry channel are retained in
# the source graph but cannot become detector nodes, so only observable-to-
# observable edges are emitted here.
OFFICIAL_BOILER_EDGES = (
    ("TK01", "LSL01", 2), ("TK01", "LSH01", 2), ("TK01", "PP04", 1),
    ("TK01", "TWIT03", 3), ("TK01", "TK04", 1), ("TK01", "SOL03", 1),
    ("TK01", "PP01A", 1), ("TK01", "PP01B", 1), ("PP04", "TK01", 1),
    ("PP04", "COND01", 1), ("TK04", "TK01", 1), ("TK04", "LSL04", 2),
    ("TK04", "LSH04", 2), ("TK04", "TWIT05", 3), ("SOL03", "TK01", 1),
    ("PP01A", "TK01", 1), ("PP01A", "PCV02", 1), ("PP01A", "HEX01T", 1),
    ("PP01B", "TK01", 1), ("PP01B", "PCV02", 1), ("PP01B", "HEX01T", 1),
    ("COND01", "TK01", 1), ("COND01", "PP04", 1), ("SOL01", "TK01", 1),
    ("PCV02", "TK01", 1), ("PCV02", "PP01A", 1), ("PCV02", "PP01B", 1),
    ("HEX01T", "PP01A", 1), ("HEX01T", "PP01B", 1), ("HEX01T", "TIT01", 3),
    ("HEX01T", "PCV01", 1), ("HEX01T", "HEX01S", 3), ("PCV01", "HEX01T", 1),
    ("PCV01", "PIT01", 2), ("PCV01", "FT01", 2), ("PCV01", "LCV01", 1),
    ("HEX01S", "HEX01T", 3), ("HEX01S", "TK02", 1), ("HEX01S", "FCV01", 1),
    ("LCV01", "PCV01", 1), ("LCV01", "TK03", 1), ("TK03", "LCV01", 1),
    ("TK03", "LIT01", 2), ("TK03", "LSH03", 2), ("TK03", "TWIT04", 3),
    ("TK03", "FCV03", 1), ("FCV03", "TK01", 1), ("FCV03", "TK03", 1),
    ("FCV03", "FT03", 2), ("HT01", "TK02", 3), ("TK02", "TIT02", 3),
    ("TK02", "PP02", 1), ("TK02", "SOL04", 1), ("TK02", "LSL02", 2),
    ("TK02", "LSH02", 2), ("TK02", "PWIT03", 2), ("PP02", "TK02", 1),
    ("PP02", "FCV02", 1), ("PP02", "FCV01", 1), ("SOL04", "TK02", 1),
    ("FCV02", "TK02", 1), ("FCV02", "PP02", 1), ("FCV01", "PP02", 1),
    ("FCV01", "PIT02", 2), ("FCV01", "FT02", 2), ("FCV01", "HEX01S", 1),
    ("SOL02", "TK02", 1),
)


def _device_name(feature: str) -> str | None:
    if not feature.startswith("P1_"):
        return None
    name = feature[3:]
    if name == "PIT01_HH":
        return "PIT01"
    for suffix in ("SP", "D", "Z", "R"):
        if name.endswith(suffix) and len(name) > len(suffix):
            candidate = name[: -len(suffix)]
            if re.search(r"\d", candidate):
                return candidate
    return name


def _add_pair(edges: set[tuple[str, str]], source: str, target: str) -> None:
    if source != target:
        edges.add(tuple(sorted((source, target))))


def _operational_context_edges(feature_names: list[str]) -> list[list[str]]:
    """Connect manual-defined operational groups without calling them physics."""

    candidate_groups = [
        ["P1_SOL01D", "P1_SOL03D"],
        ["P1_STSP", "P1_TIT01", "P1_TIT02", "P1_TIT03"],
        [
            "P2_24Vdc", "P2_ATSW_Lamp", "P2_AutoGO", "P2_AutoSD", "P2_Emerg",
            "P2_MASW", "P2_MASW_Lamp", "P2_ManualGO", "P2_ManualSD", "P2_OnOff",
            "P2_RTR", "P2_SCO", "P2_SCST", "P2_SIT01", "P2_TripEx",
        ],
        [
            "P2_VIBTR01", "P2_VIBTR02", "P2_VIBTR03", "P2_VIBTR04", "P2_VT01",
            "P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04",
        ],
        ["P3_LIT01", "P3_LH01", "P3_LL01", "P3_LCP01D", "P3_LCV01D", "P3_FIT01", "P3_PIT01"],
        ["P4_HT_PO", "P4_HT_FD", "P4_HT_PS", "P4_LD"],
        [
            "P4_ST_LD", "P4_ST_FD", "P4_ST_GOV", "P4_ST_PO", "P4_ST_PS",
            "P4_ST_PT01", "P4_ST_TT01", "P4_LD",
        ],
    ]
    available = set(feature_names)
    edges: set[tuple[str, str]] = set()
    for candidates in candidate_groups:
        group = [name for name in candidates if name in available]
        if len(group) < 2:
            continue
        anchor = group[0]
        for member in group[1:]:
            _add_pair(edges, anchor, member)
    return [list(edge) for edge in sorted(edges)]


def _learn_dependency_edges(
    feature_names: list[str], train_path: Path, top_k: int, minimum: float
) -> tuple[list[list[str]], list[list[str]]]:
    with np.load(train_path) as arrays:
        x = arrays["x"]
        if x.ndim != 4 or x.shape[2] != len(feature_names):
            raise ValueError("Training windows do not match metadata feature order")
        # Window endpoints avoid reconstructing a very large flattened matrix.
        values = np.asarray(x[:, -1, :, 0], dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = np.sqrt(np.sum(centered * centered, axis=0))
    normalized = np.divide(
        centered,
        scale[None, :],
        out=np.zeros_like(centered),
        where=scale[None, :] > 1e-12,
    )
    correlation = np.nan_to_num(
        normalized.T @ normalized, nan=0.0, posinf=0.0, neginf=0.0
    )
    np.fill_diagonal(correlation, 0.0)

    def group(name: str) -> str:
        if name.startswith("x"):
            return "P1_CONTROL"
        if name.startswith("P4_ST_"):
            return "P4_ST"
        if name.startswith("P4_HT_"):
            return "P4_HT"
        return name.split("_", 1)[0]

    groups = [group(name) for name in feature_names]
    same_process: set[tuple[str, str]] = set()
    hil: set[tuple[str, str]] = set()
    for i, name in enumerate(feature_names):
        same_candidates = [
            j for j in range(len(feature_names))
            if j != i
            and (
                groups[j] == groups[i]
                or {groups[j], groups[i]} <= {"P1", "P1_CONTROL"}
            )
        ]
        ranked = sorted(same_candidates, key=lambda j: abs(correlation[i, j]), reverse=True)
        for j in ranked[:top_k]:
            if abs(correlation[i, j]) >= minimum:
                _add_pair(same_process, name, feature_names[j])

        allowed_hil = {
            "P1": {"P2", "P4_ST"}, "P1_CONTROL": {"P2", "P4_ST"},
            "P2": {"P1", "P1_CONTROL", "P4_ST"}, "P4_ST": {"P1", "P1_CONTROL", "P2"},
            "P3": {"P4_HT"}, "P4_HT": {"P3"},
        }.get(groups[i], set())
        cross_candidates = [j for j in range(len(feature_names)) if groups[j] in allowed_hil]
        if cross_candidates:
            j = max(cross_candidates, key=lambda k: abs(correlation[i, k]))
            if abs(correlation[i, j]) >= minimum:
                _add_pair(hil, name, feature_names[j])
    return [list(edge) for edge in sorted(same_process)], [list(edge) for edge in sorted(hil)]


def build_topology(
    metadata_path: Path,
    train_path: Path,
    top_k: int,
    minimum_correlation: float,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = list(metadata["feature_names"])
    by_device: dict[str, list[str]] = {}
    for feature in feature_names:
        device = _device_name(feature)
        if device:
            by_device.setdefault(device, []).append(feature)

    relation_by_code = {1: "official_fluid", 2: "official_measurement", 3: "official_thermal"}
    official: dict[str, set[tuple[str, str]]] = {name: set() for name in relation_by_code.values()}
    for source_device, target_device, dynamics in OFFICIAL_BOILER_EDGES:
        for source in by_device.get(source_device, []):
            for target in by_device.get(target_device, []):
                if source != target:
                    official[relation_by_code[dynamics]].add((source, target))

    device_channels: set[tuple[str, str]] = set()
    for channels in by_device.values():
        if len(channels) > 1:
            for i, source in enumerate(channels):
                for target in channels[i + 1 :]:
                    _add_pair(device_channels, source, target)

    normal_edges, hil_edges = _learn_dependency_edges(
        feature_names, train_path, top_k, minimum_correlation
    )
    operational_edges = _operational_context_edges(feature_names)
    relations = {
        **{
            name: [list(edge) for edge in sorted(edges)]
            for name, edges in official.items()
            if edges
        },
        "device_channel": [list(edge) for edge in sorted(device_channels)],
        "operational_context": operational_edges,
        "normal_dependency": normal_edges,
        "hil_constrained_dependency": hil_edges,
    }
    connected = {node for edges in relations.values() for edge in edges for node in edge}
    return {
        "schema_version": 1,
        "nodes": feature_names,
        "relations": relations,
        "provenance": {
            "official_fluid/measurement/thermal": {
                "source": "https://github.com/icsdataset/hai/blob/master/graph/boiler/phy_boiler.json",
                "scope": "P1 boiler only; projected to observed HAI-23.05 telemetry channels",
            },
            "device_channel": {
                "source": "HAI-23.05 telemetry tag naming",
                "scope": "command, response, setpoint, and status channels of the same device",
            },
            "operational_context": {
                "source": "HAI technical-manual subsystem and operating-signal descriptions",
                "scope": "manual semantic groups; not claimed as authenticated physical edges",
            },
            "normal_dependency": {
                "source": str(train_path),
                "scope": "top absolute correlations learned only from normal training windows",
                "top_k": top_k,
                "minimum_absolute_correlation": minimum_correlation,
            },
            "hil_constrained_dependency": {
                "source": "HAI technical manual process coupling plus normal-training correlation",
                "scope": (
                    "P1-P2/P4 steam and P3-P4 pumped-storage HIL pairs; "
                    "not claimed as physical edges"
                ),
            },
        },
        "summary": {
            "nodes": len(feature_names),
            "connected_nodes": len(connected),
            "isolated_nodes": sorted(set(feature_names) - connected),
            "edges_by_relation": {name: len(edges) for name, edges in relations.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a provenance-separated HAI-23.05 topology")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--train-windows", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--minimum-correlation", type=float, default=0.15)
    args = parser.parse_args()
    topology = build_topology(
        Path(args.metadata), Path(args.train_windows), args.top_k, args.minimum_correlation
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(topology, indent=2), encoding="utf-8")
    print(json.dumps(topology["summary"], indent=2))
    print(f"Saved topology: {output.resolve()}")


if __name__ == "__main__":
    main()
