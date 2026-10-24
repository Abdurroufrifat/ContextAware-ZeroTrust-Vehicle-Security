"""Fixed clean-route threshold calibration; no neural model or physical truth claim."""

import numpy as np


def fit_threshold(route_scores, baseline):
    if type(baseline) not in (float, int) or not np.isfinite(baseline) or len(route_scores) != 3:
        raise ValueError("three_clean_routes_and_finite_baseline_required")
    quantiles, counts = {}, {}
    for route, values in sorted(route_scores.items()):
        x = np.asarray(values, dtype=float)
        if (
            type(route) is not str
            or not route
            or x.ndim != 1
            or len(x) < 32
            or not np.isfinite(x).all()
        ):
            raise ValueError("invalid_clean_route_scores")
        quantiles[route] = float(np.quantile(x, 0.95))
        counts[route] = len(x)
    return dict(
        threshold=max(float(baseline), *quantiles.values()),
        baseline=float(baseline),
        route_quantiles=quantiles,
        route_windows=counts,
        quantile=0.95,
        guarantee="No distribution-free future-route guarantee; dependent windows.",
    )


def check_reservations(trips, prior):
    days = {t.get("day", t.get("route_id", "").split("|")[-1][:10]) for t in prior}
    routes = {t.get("route_id") for t in prior}
    expected = ["calibration"] * 3 + ["validation"] * 2 + ["test"] * 5
    if (
        len(trips) != 10
        or [t["phase"] for t in trips] != expected
        or len({t["id"] for t in trips}) != 10
        or len({t["day"] for t in trips}) != 10
        or len({t["route_id"] for t in trips}) != 10
        or any(t["day"] in days or t["route_id"] in routes for t in trips)
    ):
        raise ValueError("ten_disjoint_fresh_dates_and_fixed_split_required")


def validation_gate(rows, cases):
    trips = sorted({r["trip"] for r in rows})
    keys = {(r["trip"], r["seed"], r["case"]) for r in rows}
    expected = {(t, s, c) for t in trips for s in [7, 19, 43] for c in cases}
    if len(trips) != 2 or "normal" not in cases or len(rows) != len(keys) or keys != expected:
        raise ValueError("complete_two_route_three_seed_validation_required")
    for row in rows:
        for field in ("baseline", "candidate", "base_grant", "baseline_grant", "candidate_grant"):
            if (
                type(row[field]) not in (float, int)
                or not np.isfinite(row[field])
                or not 0 <= row[field] <= 1
            ):
                raise ValueError("bounded_complete_validation_metrics_required")

    def mean(case, key):
        return float(
            np.mean(
                [
                    np.mean([r[key] for r in rows if r["case"] == case and r["trip"] == trip])
                    for trip in trips
                ]
            )
        )

    old, new = mean("normal", "baseline"), mean("normal", "candidate")

    def route_mean(trip, case, key):
        return float(np.mean([r[key] for r in rows if r["trip"] == trip and r["case"] == case]))

    worst_normal = max(route_mean(t, "normal", "candidate") for t in trips)
    drops = {
        case: max(route_mean(t, case, "baseline") - route_mean(t, case, "candidate") for t in trips)
        for case in cases
        if case != "normal"
    }
    checks = dict(
        normal_fpr_at_most_5_percent=worst_normal <= 0.05 + 1e-12,
        at_least_1pp_fpr_reduction=old - new >= 0.01 - 1e-12,
        normal_grants_not_lower=mean("normal", "candidate_grant")
        >= mean("normal", "baseline_grant") - 1e-12,
        each_attack_tpr_loss_at_most_5pp=all(v <= 0.05 + 1e-12 for v in drops.values()),
    )
    return dict(
        passed=all(checks.values()),
        checks=checks,
        baseline_normal_fpr=old,
        candidate_normal_fpr=new,
        candidate_worst_route_normal_fpr=worst_normal,
        attack_tpr_drop=drops,
        status="VALIDATION_PASSED" if all(checks.values()) else "NOT_PROMOTED_TEST_LOCKED",
    )
