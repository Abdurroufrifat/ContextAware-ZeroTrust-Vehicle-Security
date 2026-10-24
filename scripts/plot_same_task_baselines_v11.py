from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


LABELS = {
    "legacy_structural": "Structural",
    "legacy_plus_phrase_policy": "Structural + phrase policy",
    "token_jaccard": "Token Jaccard",
    "word_tfidf": "Word TF IDF",
    "character_tfidf": "Character TF IDF",
    "canonical_surface": "Canonical surface",
    "registered_surface": "Registered surfaces",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot V11 same task verifier baselines")
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    heldout = metrics["by_split"]["held_out"]["baselines"]
    order = metrics["baseline_order"]
    colors = ["#7A7A7A", "#E69F00", "#56B4E9", "#0072B2", "#009E73", "#CC79A7", "#D55E00"]

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.4), constrained_layout=True)
    y = list(range(len(order)))
    safe = [heldout[name]["safe_coverage"]["rate"] for name in order]
    contained = [heldout[name]["attack_containment"]["rate"] for name in order]
    safe_ci = [heldout[name]["safe_coverage"]["rate_95pct_wilson"] for name in order]
    containment_ci = [
        heldout[name]["attack_containment"]["rate_95pct_wilson"] for name in order
    ]

    def errors(values, intervals):
        return [
            [value - interval[0] for value, interval in zip(values, intervals)],
            [interval[1] - value for value, interval in zip(values, intervals)],
        ]

    left.errorbar(
        safe,
        y,
        xerr=errors(safe, safe_ci),
        fmt="none",
        ecolor="#555555",
        capsize=3,
        linewidth=1,
        zorder=1,
    )
    left.scatter(safe, y, c=colors, s=55, zorder=2)
    left.set_title("Safe explanation coverage")
    left.set_xlabel("Acceptance rate with 95% Wilson interval")
    left.set_xlim(-0.03, 1.03)
    left.set_yticks(y, [LABELS[name] for name in order])
    left.grid(axis="x", alpha=0.25)

    right.errorbar(
        contained,
        y,
        xerr=errors(contained, containment_ci),
        fmt="none",
        ecolor="#555555",
        capsize=3,
        linewidth=1,
        zorder=1,
    )
    right.scatter(contained, y, c=colors, s=55, zorder=2)
    right.set_title("Adversarial explanation containment")
    right.set_xlabel("Rejection rate with 95% Wilson interval")
    right.set_xlim(-0.03, 1.03)
    right.set_yticks(y, [])
    right.grid(axis="x", alpha=0.25)

    fig.suptitle("Heldout same task verification baselines", fontsize=13)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        destination = args.output_prefix.with_suffix(f".{suffix}")
        fig.savefig(destination, dpi=300, bbox_inches="tight")
        print(f"Saved {destination.resolve()}")
    plt.close(fig)


if __name__ == "__main__":
    main()

