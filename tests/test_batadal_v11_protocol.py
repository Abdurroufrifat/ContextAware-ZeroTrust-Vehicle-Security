from pathlib import Path

import yaml

from scripts.make_secondary_seed_configs_v11 import safe_slug


ROOT = Path(__file__).resolve().parents[1]


def test_batadal_config_freezes_independent_labeled_split() -> None:
    config = yaml.safe_load((ROOT / "configs/batadal_v11.yaml").read_text())
    dataset = config["dataset"]
    assert dataset["train_csvs"] == ["data/raw/batadal/BATADAL_dataset03.csv"]
    assert dataset["test_csvs"] == ["data/raw/batadal/BATADAL_dataset04.csv"]
    assert dataset["label_column"] == "ATT_FLAG"
    assert -999 in dataset["normal_values"]
    assert 0 in dataset["normal_values"]
    assert len(dataset["features"]) == 43
    assert dataset["window_size"] == 24
    assert config["evaluation"]["seeds"] == [7, 17, 29, 43, 61]


def test_secondary_artifact_slug_is_safe() -> None:
    assert safe_slug("BATADAL V11") == "batadal_v11"
