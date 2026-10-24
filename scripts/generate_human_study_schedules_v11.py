from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


CONDITIONS = ("canonical", "adversarial", "protected")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate balanced V11 human decision study schedules"
    )
    parser.add_argument("--case-bank", required=True, type=Path)
    parser.add_argument("--participants", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.participants < 3 or args.participants % 3:
        raise ValueError("participants must be at least three and divisible by three")

    bank = json.loads(args.case_bank.read_text(encoding="utf-8"))
    case_ids = [str(case["case_id"]) for case in bank["cases"]]
    if len(case_ids) % 3:
        raise ValueError("The number of cases must be divisible by three")

    schedules = []
    balance = Counter()
    for participant_index in range(args.participants):
        rotation = participant_index % len(CONDITIONS)
        trials = [
            {
                "case_id": case_id,
                "condition": CONDITIONS[(case_index + rotation) % len(CONDITIONS)],
            }
            for case_index, case_id in enumerate(case_ids)
        ]
        rng = random.Random(args.seed + participant_index)
        rng.shuffle(trials)
        for position, trial in enumerate(trials, start=1):
            trial["position"] = position
            balance[(trial["case_id"], trial["condition"])] += 1
        schedules.append(
            {
                "allocation_slot": participant_index + 1,
                "participant_code_template": f"OP{participant_index + 1:03d}",
                "rotation": rotation,
                "trials": trials,
            }
        )

    expected = args.participants // len(CONDITIONS)
    if any(count != expected for count in balance.values()):
        raise AssertionError("Case by condition allocation is not balanced")
    payload = {
        "protocol_version": "cps-truthframe-human-decision-schedules-v11.0",
        "status": "ALLOCATION_ONLY_DO_NOT_RECRUIT_BEFORE_APPROVAL",
        "seed": args.seed,
        "participants": args.participants,
        "cases_per_participant": len(case_ids),
        "conditions": list(CONDITIONS),
        "case_condition_count": expected,
        "schedules": schedules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "seed", "participants", "cases_per_participant", "case_condition_count"
    )}, indent=2))


if __name__ == "__main__":
    main()

