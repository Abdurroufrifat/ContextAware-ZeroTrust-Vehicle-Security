"""Bounded local vehicle-data inventory and Stage 1 evidence preservation.

Run from D:\\CO-ACT-CPS. No downloads, data edits, or vehicle-model training.
Headers identify candidates only; no dataset is approved from a sampled schema.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath

PROBE_BYTES = 32768
SAMPLE_ROWS = 20
REQUIRED_FLAT_FIELDS = (
    "drive_id",
    "vehicle_id",
    "frame_id",
    "timestamp_ms",
    "subject_id",
    "credential_valid",
    "revoked",
    "attestation_ok",
    "gnss_speed_mps",
    "imu_acceleration_mps2",
    "imu_yaw_rate_rps",
    "wheel_speed_mps",
    "wheel_acceleration_mps2",
    "wheel_yaw_rate_rps",
    "can_anomaly_score",
    "v2x_inconsistency_score",
    "v2x_uncertainty",
)
DOMAINS = {
    "gnss": [{"gnssspeedmps"}, {"gnsslatitude", "gnsslongitude"}, {"gnsslat", "gnsslon"}],
    "imu": [{"imuaccelerationmps2", "imuyawraterps"}, {"imuax", "imuay", "imugyroz"}],
    "wheel": [{"wheelspeedmps"}, {"odometryspeedmps"}],
    "can": [{"cananomalyscore"}, {"canid", "candlc"}, {"id", "dlc", "data"}],
    "identity": [
        {"subjectid", "credentialvalid"},
        {"identitysubjectid", "identitycredentialvalid"},
    ],
    "v2x": [{"v2xinconsistencyscore", "v2xuncertainty"}, {"v2xsenderid", "v2xmessagetimestampms"}],
}
RANGES = {
    "gnss_speed_mps": (0, 100),
    "wheel_speed_mps": (0, 100),
    "imu_acceleration_mps2": (-30, 30),
    "wheel_acceleration_mps2": (-30, 30),
    "imu_yaw_rate_rps": (-10, 10),
    "wheel_yaw_rate_rps": (-10, 10),
    "can_anomaly_score": (0, 1),
    "v2x_inconsistency_score": (0, 1),
    "v2x_uncertainty": (0, 1),
}
ELIGIBLE_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".npz",
    ".npy",
    ".mat",
    ".bag",
    ".db3",
    ".pcap",
    ".pcapng",
    ".mf4",
    ".blf",
    ".zip",
    ".gz",
}


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def flatten(value, prefix="", depth=0):
    if type(value) is not dict or depth > 3:
        return {}
    result = {}
    for key, item in value.items():
        name = prefix + str(key)
        if type(item) is dict:
            result.update(flatten(item, name + ".", depth + 1))
        elif not isinstance(item, (list, dict)):
            result[name] = item
    return result


def sample_issues(columns, rows):
    issues = set()
    if len(set(columns)) != len(columns):
        issues.add("duplicate_column_names")
    if not set(REQUIRED_FLAT_FIELDS).issubset(columns):
        return sorted(issues)
    seen, last_times = set(), {}
    for row in rows:
        if not isinstance(row, dict):
            issues.add("non_object_sample_row")
            continue
        if any(row.get(k) is None or str(row[k]).strip() == "" for k in REQUIRED_FLAT_FIELDS):
            issues.add("missing_required_sample_value")
        for name, (low, high) in RANGES.items():
            try:
                if isinstance(row.get(name), bool):
                    raise ValueError
                value = float(row.get(name))
                if not math.isfinite(value):
                    issues.add("nonfinite:" + name)
                elif not low <= value <= high:
                    issues.add("out_of_range:" + name)
            except (ValueError, TypeError, OverflowError):
                issues.add("nonnumeric:" + name)
        for name in ("credential_valid", "revoked", "attestation_ok"):
            if str(row.get(name)).lower() not in {"true", "false", "0", "1"}:
                issues.add("invalid_boolean_encoding:" + name)
        key = tuple(str(row.get(k)) for k in ("drive_id", "vehicle_id", "frame_id"))
        if key in seen:
            issues.add("duplicate_drive_vehicle_frame")
        seen.add(key)
        try:
            if isinstance(row.get("timestamp_ms"), bool):
                raise ValueError
            timestamp = float(row.get("timestamp_ms"))
            if not math.isfinite(timestamp) or timestamp < 0 or not timestamp.is_integer():
                raise ValueError
            session = key[:2]
            if timestamp < last_times.get(session, -1):
                issues.add("timestamp_decreases_within_drive")
            last_times[session] = timestamp
        except (ValueError, TypeError, OverflowError):
            issues.add("invalid_integer_timestamp_ms")
    return sorted(issues)


def _looks_like_data_row(columns):
    numeric = 0
    for value in columns:
        try:
            float(value)
            numeric += 1
        except ValueError:
            if re.fullmatch(r"0x[0-9a-fA-F]+", value):
                numeric += 1
    return bool(columns) and numeric / len(columns) >= 0.6


def inspect_file(path, root):
    path, root = Path(path), Path(root)
    stat = path.stat()
    result = {
        "path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "schema_status": "container_requires_explicit_adapter",
        "columns": [],
        "vehicle_domains": [],
        "full_flat_schema_candidate": False,
        "quality_scope": "bounded schema/sample check only",
        "sample_rows_checked": 0,
        "sample_issues": [],
    }
    columns, rows = [], []
    try:
        if path.name.lower().endswith((".csv.gz", ".tsv.gz", ".jsonl.gz", ".ndjson.gz")):
            with gzip.open(path, "rb") as stream:
                probe = stream.read(PROBE_BYTES)
            name = path.name[:-3].lower()
            result["probe_scope"] = "first decompressed bytes only"
        else:
            with path.open("rb") as stream:
                probe = stream.read(PROBE_BYTES)
            name = path.name.lower()
            result["probe_scope"] = "first file bytes only"
        result["schema_sample_sha256"] = hashlib.sha256(probe).hexdigest()
        result["schema_sample_bytes"] = len(probe)
        # Drop an incomplete trailing CSV/JSONL row from a bounded probe.
        if len(probe) == PROBE_BYTES and b"\n" in probe:
            probe = probe[: probe.rfind(b"\n") + 1]
        if name.endswith((".csv", ".tsv")):
            text = probe.decode("utf-8-sig")
            delimiter = "\t" if name.endswith(".tsv") else ","
            try:
                delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",\t;").delimiter
            except csv.Error:
                pass
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            columns = [c.strip() for c in next(reader)]
            if len(columns) > 512 or _looks_like_data_row(columns):
                result["schema_status"] = "headerless_or_ambiguous"
                return result
            for i, values in enumerate(reader):
                if i >= SAMPLE_ROWS:
                    break
                if len(values) != len(columns):
                    result["sample_issues"].append("sample_column_count_mismatch")
                rows.append(dict(zip(columns, values, strict=False)))
            result["schema_status"] = "header_read"
        elif name.endswith((".jsonl", ".ndjson")):
            for line in probe.decode("utf-8-sig").splitlines():
                if line.strip():
                    value = json.loads(line)
                    rows.append(flatten(value))
                    if len(rows) == SAMPLE_ROWS:
                        break
            columns = sorted({key for row in rows for key in row})
            result["schema_status"] = "object_keys_read"
        elif name.endswith(".json"):
            if len(probe) == PROBE_BYTES:
                result["schema_status"] = "large_json_requires_explicit_adapter"
                return result
            value = json.loads(probe.decode("utf-8-sig"))
            if isinstance(value, list):
                rows = [flatten(v) for v in value[:SAMPLE_ROWS] if isinstance(v, dict)]
            elif isinstance(value, dict):
                rows = [flatten(value)]
            columns = sorted({key for row in rows for key in row})
            result["schema_status"] = "object_keys_read"
        names = {normalized(c) for c in columns}
        result["columns"] = columns[:512]
        result["vehicle_domains"] = sorted(
            domain
            for domain, variants in DOMAINS.items()
            if any(fields <= names for fields in variants)
        )
        result["full_flat_schema_candidate"] = set(REQUIRED_FLAT_FIELDS) <= set(columns)
        result["sample_rows_checked"] = len(rows)
        result["sample_issues"] = sorted(
            set(result["sample_issues"] + sample_issues(columns, rows))
        )
    except (OSError, UnicodeError, ValueError, StopIteration, csv.Error, RecursionError) as exc:
        result["schema_status"] = "unreadable_schema"
        result["error_type"] = type(exc).__name__
    return result


def audit_data(root, max_files=2000):
    root = Path(root).resolve()
    data = root / "data"
    report = {
        "scan_root": str(data),
        "files": [],
        "scan_complete": True,
        "full_flat_schema_candidates": [],
        "observed_vehicle_domains": [],
        "excluded_links_or_outside_paths": [],
        "uninspectable_paths": [],
        "max_files": max_files,
        "sample_rows_per_file_limit": SAMPLE_ROWS,
        "schema_probe_bytes_limit": PROBE_BYTES,
        "data_files_modified": False,
        "alignment_performed": False,
        "dataset_approved": False,
    }
    if not data.is_dir():
        report.update(scan_complete=False, decision="DATA_DIRECTORY_MISSING")
        return report
    if not data.resolve().is_relative_to(root):
        report.update(scan_complete=False, decision="DATA_PATH_OUTSIDE_PROJECT")
        return report

    def walk_error(error):
        report["uninspectable_paths"].append(str(error.filename))
        report["scan_complete"] = False

    truncated = False
    for directory, dirs, files in os.walk(data, followlinks=False, onerror=walk_error):
        dirs.sort()
        for d in list(dirs):
            candidate = Path(directory) / d
            if candidate.is_symlink() or not candidate.resolve().is_relative_to(data):
                dirs.remove(d)
                report["excluded_links_or_outside_paths"].append(str(candidate.relative_to(root)))
                report["scan_complete"] = False
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() not in ELIGIBLE_SUFFIXES:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(data):
                report["excluded_links_or_outside_paths"].append(str(path.relative_to(root)))
                report["scan_complete"] = False
                continue
            if len(report["files"]) >= max_files:
                truncated = True
                break
            try:
                report["files"].append(inspect_file(path, root))
            except OSError as exc:
                report["uninspectable_paths"].append(str(exc.filename))
                report["scan_complete"] = False
        if truncated:
            break
    if truncated:
        report["scan_complete"] = False
    domains = sorted({d for f in report["files"] for d in f["vehicle_domains"]})
    candidates = [f["path"] for f in report["files"] if f["full_flat_schema_candidate"]]
    report["observed_vehicle_domains"] = domains
    report["full_flat_schema_candidates"] = candidates
    report["eligible_files_inspected"] = len(report["files"])
    report["decision"] = (
        "SCHEMA_CANDIDATES_REQUIRE_FULL_DATA_VALIDATION"
        if candidates
        else "PARTIAL_VEHICLE_SOURCES_REQUIRE_MAPPING_AND_ALIGNMENT"
        if domains
        else "NO_VEHICLE_SCHEMA_IDENTIFIED_WITHIN_SCANNED_FILES"
    )
    report["qualification"] = (
        "File headers and limited samples identify possible sources only. "
        "A union of domains across files does not establish shared clocks, sessions, vehicle IDs, "
        "source independence, units, attack coverage, licensing or native authentication."
    )
    return report


def source_path(root, recorded):
    if not isinstance(recorded, str):
        raise ValueError("source_reference_outside_project")
    root = Path(root).resolve()
    if Path(recorded).is_absolute():
        try:
            native = Path(recorded).resolve().relative_to(root)
            relative = PureWindowsPath(*native.parts)
        except ValueError:
            # Historical Windows manifests remain relocatable under the explicit
            # D:/CO-ACT-CPS prefix check below, including on POSIX hosts.
            relative = PureWindowsPath(recorded)
    else:
        relative = PureWindowsPath(recorded)
    if relative.is_absolute():
        try:
            relative = relative.relative_to(PureWindowsPath("D:/CO-ACT-CPS"))
        except ValueError as exc:
            raise ValueError("source_reference_outside_project") from exc
    if (
        not relative.parts
        or any(p in {"..", "."} or ":" in p for p in relative.parts)
        or relative.parts[0] not in {"src", "scripts", "configs", "tests"}
    ):
        raise ValueError("source_reference_outside_project")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source_reference_outside_project")
    return path


def freeze_stage1(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("output_must_remain_inside_project")
    source_dir = root / "artifacts/vehicle_zero_trust_stage1"
    metrics_path = source_dir / "metrics.json"
    result = {
        "metrics_found": metrics_path.is_file(),
        "source_checks": [],
        "all_recorded_sources_match": False,
        "evidence_copy": None,
    }
    if not metrics_path.is_file():
        return result
    if not metrics_path.resolve().is_relative_to(root):
        result["error"] = "stage1_metrics_path_outside_project"
        return result
    try:
        metrics_bytes = metrics_path.read_bytes()
        metrics = json.loads(metrics_bytes.decode("utf-8-sig"))
        if type(metrics) is not dict or type(metrics.get("source_sha256", {})) is not dict:
            raise ValueError("invalid_stage1_metrics")
    except (OSError, ValueError, UnicodeError):
        result["error"] = "invalid_stage1_metrics"
        return result
    snapshot = output / "stage1_evidence"
    snapshot.mkdir(parents=True, exist_ok=False)
    (snapshot / "metrics.json").write_bytes(metrics_bytes)
    for name in (
        "contract_cases.csv",
        "policy_card_mutations.csv",
        "explanation_mutations.csv",
        "decision_examples.json",
    ):
        path = source_dir / name
        if path.is_file() and path.resolve().is_relative_to(root):
            shutil.copy2(path, snapshot / name)
    for recorded, expected in metrics.get("source_sha256", {}).items():
        check = {"recorded_path": recorded, "expected_sha256": expected, "matched": False}
        try:
            path = source_path(root, recorded)
            content = path.read_bytes()
            actual = hashlib.sha256(content).hexdigest()
            check.update(actual_sha256=actual, matched=actual == expected)
            destination = snapshot / "source_snapshot" / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        except (OSError, ValueError) as exc:
            check["error_type"] = type(exc).__name__
        result["source_checks"].append(check)
    result["all_recorded_sources_match"] = bool(result["source_checks"]) and all(
        c["matched"] for c in result["source_checks"]
    )
    result["evidence_copy"] = str(snapshot.relative_to(root))
    ledger = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            ledger.append(
                hashlib.sha256(path.read_bytes()).hexdigest()
                + "  "
                + path.relative_to(snapshot).as_posix()
            )
    (snapshot / "SHA256SUMS.txt").write_text("\n".join(ledger) + "\n", encoding="utf-8")
    return result


def environment_summary():
    result = {
        "python": sys.version.split()[0],
        "carla_client_module_found": importlib.util.find_spec("carla") is not None,
        "carla_server_compatibility": "not_checked",
        "installed_versions": {},
    }
    for name in ("torch", "numpy", "carla"):
        try:
            result["installed_versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["installed_versions"][name] = None
    executable = shutil.which("nvidia-smi")
    result["nvidia_gpu_query"] = "nvidia-smi not found"
    if executable:
        try:
            process = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            result["nvidia_gpu_query"] = (
                process.stdout.strip()[:1000] if process.returncode == 0 else "query failed"
            )
        except (OSError, subprocess.TimeoutExpired):
            result["nvidia_gpu_query"] = "query failed or timed out"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"D:\CO-ACT-CPS")
    parser.add_argument("--max-files", type=int, default=2000)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir() or not 1 <= args.max_files <= 100000:
        parser.error("existing project root and max-files between 1 and 100000 required")
    output_root = root / "artifacts/vehicle_zero_trust_stage2_readiness"
    if not output_root.resolve().is_relative_to(root):
        parser.error("artifact output path must remain inside the project root")
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
    output = output_root / run_id
    output.mkdir()
    report = {
        "stage": "vehicle-data-readiness-only",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "stage1": freeze_stage1(root, output),
        "data_inventory": audit_data(root, args.max_files),
        "environment": environment_summary(),
        "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "new_vehicle_experiments_performed": False,
        "output_directory": str(output.relative_to(root)),
    }
    text = json.dumps(report, indent=2, ensure_ascii=True)
    (output / "readiness.json").write_text(text, encoding="utf-8")
    (output_root / "latest_readiness.json").write_text(text, encoding="utf-8")
    summary = {
        "stage1_source_hashes_match": report["stage1"]["all_recorded_sources_match"],
        "files_inspected": report["data_inventory"].get("eligible_files_inspected", 0),
        "scan_complete": report["data_inventory"]["scan_complete"],
        "data_decision": report["data_inventory"]["decision"],
        "candidate_domains": report["data_inventory"]["observed_vehicle_domains"],
        "report": str(output_root / "latest_readiness.json"),
    }
    print(json.dumps(summary, indent=2))
    if report["stage1"]["metrics_found"] and not report["stage1"]["all_recorded_sources_match"]:
        print(
            "Stage 1 hash verification did not pass. "
            "Inspect this report before new experiments."
        )


if __name__ == "__main__":
    main()
