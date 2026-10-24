"""Evaluate frozen classical baselines on the official BATADAL final test set.

Thresholds and score scaling are determined from the normal validation split only.
The official final attack intervals are used only after all baseline choices are fixed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 17, 29, 43, 61]
INTERVALS = [
    (8, "2017-01-16 09:00", "2017-01-19 06:00", 70),
    (9, "2017-01-30 08:00", "2017-02-02 00:00", 65),
    (10, "2017-02-09 03:00", "2017-02-10 09:00", 31),
    (11, "2017-02-12 01:00", "2017-02-13 07:00", 31),
    (12, "2017-02-24 05:00", "2017-02-28 08:00", 100),
    (13, "2017-03-10 14:00", "2017-03-13 21:00", 80),
    (14, "2017-03-25 20:00", "2017-03-27 01:00", 30),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def expected_calibration_error(y_true: np.ndarray, probability: np.ndarray) -> float:
    boundaries = np.linspace(0.0, 1.0, 11)
    result = 0.0
    for index in range(10):
        if index == 9:
            selected = (probability >= boundaries[index]) & (probability <= boundaries[index + 1])
        else:
            selected = (probability >= boundaries[index]) & (probability < boundaries[index + 1])
        if selected.any():
            result += float(selected.mean()) * abs(
                float(y_true[selected].mean()) - float(probability[selected].mean())
            )
    return result


def binary_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    normal = y_true == 0
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y_true, probability)),
        "ece": expected_calibration_error(y_true, probability),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "false_alarm_rate": float(prediction[normal].mean()),
    }


def positive_run_count(values: np.ndarray) -> int:
    binary = np.asarray(values, dtype=int)
    return int(np.sum(np.diff(np.pad(binary, (1, 1))) == 1))


def operating_metrics(
    y_true: np.ndarray, probability: np.ndarray, episodes: list[dict]
) -> dict[str, float]:
    alarms = probability >= 0.5
    detected_delays = [row["delay_hours"] for row in episodes if row["detected"]]
    return {
        "attack_episode_recall": float(np.mean([row["detected"] for row in episodes])),
        "mean_delay_detected_episodes_hours": float(np.mean(detected_delays)),
        "false_alarm_episode_count": float(positive_run_count(alarms & (y_true == 0))),
    }


def calibrate(validation_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    threshold = float(np.quantile(validation_scores, 0.995))
    lower = float(np.quantile(validation_scores, 0.95))
    temperature = max(threshold - lower, float(np.std(validation_scores)) * 0.25, 1e-8)
    logits = np.clip((test_scores - threshold) / temperature, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-logits)), threshold, temperature


def persistence_scores(x: np.ndarray) -> np.ndarray:
    residual = np.mean(np.abs(x[:, -1] - x[:, -2]), axis=-1)
    top_count = max(1, int(round(residual.shape[1] * 0.10)))
    return np.sort(residual, axis=1)[:, -top_count:].mean(axis=1)


def summary_features(x: np.ndarray) -> np.ndarray:
    values = x[..., 0]
    return np.concatenate(
        (values[:, -1], values.mean(axis=1), values.std(axis=1), values[:, -1] - values[:, 0]),
        axis=1,
    ).astype(np.float32)


def make_windows(values: np.ndarray, labels: np.ndarray, size: int = 24):
    count = len(values) - size + 1
    x = np.stack([values[start : start + size] for start in range(count)]).astype(np.float32)
    ends = np.arange(size - 1, len(values))
    hourly = labels[ends]
    any_label = np.asarray([labels[start : start + size].max() for start in range(count)])
    return x[..., None], ends, hourly, any_label


def episode_results(times: pd.Series, ends: np.ndarray, probability: np.ndarray):
    alarms = probability >= 0.5
    rows = []
    for attack_id, start, end, _ in INTERVALS:
        selected = times.iloc[ends].between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
        hits = np.flatnonzero(selected & alarms)
        delay = None
        if len(hits):
            delay = (times.iloc[ends[hits[0]]] - pd.Timestamp(start)).total_seconds() / 3600.0
        rows.append({"attack_id": attack_id, "detected": bool(len(hits)), "delay_hours": delay})
    return rows


def summarize(records: list[dict], field: str) -> dict[str, dict[str, float]]:
    keys = list(records[0][field])
    return {
        key: {
            "mean": float(np.mean([record[field][key] for record in records])),
            "sample_standard_deviation": float(np.std([record[field][key] for record in records], ddof=1)),
        }
        for key in keys
    }


def main() -> None:
    data = ROOT / "inputs"
    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    source = data / "BATADAL_test_dataset.csv"
    metadata_path = data / "metadata.json"
    train_path = data / "train.npz"
    validation_path = data / "val.npz"
    proposed_path = data / "final_results.json"
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    proposed = json.loads(proposed_path.read_text(encoding="utf-8"))

    frame = pd.read_csv(source)
    frame.columns = frame.columns.str.strip()
    times = pd.to_datetime(frame["DATETIME"], format="%d/%m/%y %H")
    assert len(frame) == 2089 and not times.duplicated().any()
    assert (times.diff().dropna() == pd.Timedelta(hours=1)).all()
    labels = np.zeros(len(frame), dtype=np.int64)
    for _, start, end, expected in INTERVALS:
        selected = times.between(pd.Timestamp(start), pd.Timestamp(end)).to_numpy()
        assert int(selected.sum()) == expected
        labels[selected] = 1
    assert int(labels.sum()) == 407

    raw = frame[meta["feature_names"]].to_numpy(dtype=np.float32)
    raw -= np.asarray(meta["scaler_mean"])
    raw /= np.asarray(meta["scaler_scale"])
    final_x, ends, hourly_y, any_y = make_windows(raw, labels)
    train = np.load(train_path, allow_pickle=False)
    validation = np.load(validation_path, allow_pickle=False)
    assert np.all(train["y"] == 0) and np.all(validation["y"] == 0)

    persistence_validation = persistence_scores(validation["x"])
    persistence_test = persistence_scores(final_x)
    probability, threshold, temperature = calibrate(persistence_validation, persistence_test)
    persistence_episodes = episode_results(times, ends, probability)
    persistence = {
        "calibration_threshold": threshold,
        "calibration_temperature": temperature,
        "hourly_metrics": binary_metrics(hourly_y, probability),
        "window_any_metrics": binary_metrics(any_y, probability),
        "episodes": persistence_episodes,
        "operating_metrics": operating_metrics(hourly_y, probability, persistence_episodes),
    }
    pd.DataFrame(
        {
            "timestamp": times.iloc[ends].to_numpy(),
            "hourly_label": hourly_y,
            "window_any_label": any_y,
            "probability": probability,
            "alarm": (probability >= 0.5).astype(int),
        }
    ).to_csv(output / "persistence_predictions.csv", index=False)

    train_features = summary_features(train["x"])
    validation_features = summary_features(validation["x"])
    final_features = summary_features(final_x)
    isolation_runs = []
    for seed in SEEDS:
        model = IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(train_features)
        validation_scores = -model.decision_function(validation_features)
        test_scores = -model.decision_function(final_features)
        probability, threshold, temperature = calibrate(validation_scores, test_scores)
        isolation_episodes = episode_results(times, ends, probability)
        record = {
            "seed": seed,
            "calibration_threshold": threshold,
            "calibration_temperature": temperature,
            "hourly_metrics": binary_metrics(hourly_y, probability),
            "window_any_metrics": binary_metrics(any_y, probability),
            "episodes": isolation_episodes,
            "operating_metrics": operating_metrics(hourly_y, probability, isolation_episodes),
        }
        isolation_runs.append(record)
        pd.DataFrame(
            {
                "timestamp": times.iloc[ends].to_numpy(),
                "hourly_label": hourly_y,
                "window_any_label": any_y,
                "probability": probability,
                "alarm": (probability >= 0.5).astype(int),
            }
        ).to_csv(output / f"isolation_forest_seed_{seed}_predictions.csv", index=False)

    proposed_runs = proposed["runs"]
    proposed_hourly_summary = summarize(proposed_runs, "hourly_metrics")
    isolation_hourly_summary = summarize(isolation_runs, "hourly_metrics")
    comparison_rows = []
    for method, summary in (
        ("CO-ACT-CPS", proposed_hourly_summary),
        ("Isolation Forest", isolation_hourly_summary),
    ):
        row = {"method": method, "runs": 5}
        for metric in ("precision", "recall", "f1", "roc_auc", "average_precision", "false_alarm_rate"):
            if metric == "false_alarm_rate" and metric not in summary:
                values = []
                for seed in SEEDS:
                    prediction_frame = pd.read_csv(data / f"seed_{seed}_predictions.csv")
                    normal = prediction_frame["hourly_label"] == 0
                    values.append(float(prediction_frame.loc[normal, "alarm"].mean()))
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            else:
                row[f"{metric}_mean"] = summary[metric]["mean"]
                row[f"{metric}_sd"] = summary[metric]["sample_standard_deviation"]
        comparison_rows.append(row)
    pmetrics = persistence["hourly_metrics"]
    comparison_rows.append(
        {
            "method": "Persistence",
            "runs": 1,
            **{f"{metric}_mean": pmetrics[metric] for metric in (
                "precision", "recall", "f1", "roc_auc", "average_precision", "false_alarm_rate"
            )},
            **{f"{metric}_sd": None for metric in (
                "precision", "recall", "f1", "roc_auc", "average_precision", "false_alarm_rate"
            )},
        }
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "final_baseline_comparison.csv", index=False)

    operational_rows = []
    for method, runs in (("CO-ACT-CPS", proposed_runs), ("Isolation Forest", isolation_runs)):
        values = []
        for record in runs:
            if method == "CO-ACT-CPS":
                prediction_frame = pd.read_csv(data / f"seed_{record['seed']}_predictions.csv")
                probability = prediction_frame["probability"].to_numpy()
                labels_for_run = prediction_frame["hourly_label"].to_numpy()
                values.append(operating_metrics(labels_for_run, probability, record["episodes"]))
            else:
                values.append(record["operating_metrics"])
        row = {"method": method, "runs": len(values)}
        for metric in values[0]:
            metric_values = [item[metric] for item in values]
            row[f"{metric}_mean"] = float(np.mean(metric_values))
            row[f"{metric}_sd"] = float(np.std(metric_values, ddof=1))
        operational_rows.append(row)
    operational_rows.append(
        {
            "method": "Persistence",
            "runs": 1,
            **{f"{key}_mean": value for key, value in persistence["operating_metrics"].items()},
            **{f"{key}_sd": None for key in persistence["operating_metrics"]},
        }
    )
    pd.DataFrame(operational_rows).to_csv(
        output / "final_baseline_episode_comparison.csv", index=False
    )

    result = {
        "protocol": "BATADAL final matched baseline evaluation v11",
        "claim_boundary": (
            "Thresholds and calibration use the normal validation split only. The final test is used "
            "once for descriptive comparison. Results do not establish superiority across plants."
        ),
        "inputs": {
            "test_sha256": digest(source),
            "metadata_sha256": digest(metadata_path),
            "train_npz_sha256": digest(train_path),
            "validation_npz_sha256": digest(validation_path),
            "proposed_results_sha256": digest(proposed_path),
        },
        "evaluated_hours": len(hourly_y),
        "attack_hours": int(hourly_y.sum()),
        "normal_hours": int((hourly_y == 0).sum()),
        "calibration_quantile": 0.995,
        "persistence": persistence,
        "isolation_forest_runs": isolation_runs,
        "isolation_forest_hourly_summary": isolation_hourly_summary,
        "proposed_hourly_summary": proposed_hourly_summary,
        "comparison": comparison_rows,
        "operational_comparison": operational_rows,
    }
    (output / "final_baseline_results.json").write_text(
        json.dumps(clean(result), indent=2, allow_nan=False), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(f"Saved matched baseline results to {output}")


if __name__ == "__main__":
    main()
