from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from coact_cps.config import load_config


def resolve(path_value: str, root: Path) -> str:
    path = Path(path_value)
    return str((root / path).resolve() if not path.is_absolute() else path.resolve())


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    if not slug:
        raise ValueError("The secondary dataset needs a nonempty artifact slug")
    return slug


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze one secondary dataset config per seed")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()

    source = args.config.resolve()
    root = source.parent.parent
    config = load_config(source)
    seeds = [int(seed) for seed in config.get("evaluation", {}).get("seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("Configure at least three distinct evaluation seeds")
    slug = safe_slug(
        str(
            config.get("evaluation", {}).get("artifact_slug")
            or config["dataset"].get("name")
            or "secondary"
        )
    )

    config["dataset"]["output_directory"] = resolve(
        str(config["dataset"]["output_directory"]), root
    )
    topology_file = config.get("topology", {}).get("file")
    if topology_file:
        config["topology"]["file"] = resolve(str(topology_file), root)
    base_artifacts = Path(config.get("artifacts", {}).get("directory", "artifacts/swat_v11"))
    if not base_artifacts.is_absolute():
        base_artifacts = (root / base_artifacts).resolve()

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        derived = yaml.safe_load(yaml.safe_dump(config, sort_keys=False))
        derived["training"]["seed"] = seed
        derived["artifacts"]["directory"] = str(base_artifacts / f"seed_{seed}")
        path = output / f"{slug}_seed_{seed}.yaml"
        path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
        print(path.resolve())


if __name__ == "__main__":
    main()
