from __future__ import annotations

import argparse
import json

from .data.csv_adapter import inspect_csv
from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coact-cps")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="Run the synthetic end-to-end smoke test")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--epochs", type=int, default=5)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--n-steps", type=int, default=1800)
    demo.add_argument("--generator", choices=("template", "hf"), default="template")
    demo.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    demo.add_argument(
        "--detection-mode",
        choices=("normal_only", "supervised"),
        default="normal_only",
    )

    inspect = subparsers.add_parser("inspect-csv", help="Print CSV columns and sample rows")
    inspect.add_argument("path")
    inspect.add_argument("--rows", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        result = run_demo(
            args.output,
            epochs=args.epochs,
            seed=args.seed,
            generator_mode=args.generator,
            model_id=args.model_id,
            n_steps=args.n_steps,
            detection_mode=args.detection_mode,
        )
        print(json.dumps(result["controlled_attack_check"], indent=2))
        print(f"Saved smoke-test artifacts in {args.output}")
    elif args.command == "inspect-csv":
        print(json.dumps(inspect_csv(args.path, args.rows), indent=2))


if __name__ == "__main__":
    main()
