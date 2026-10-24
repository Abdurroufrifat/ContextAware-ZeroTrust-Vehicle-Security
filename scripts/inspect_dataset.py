from __future__ import annotations

import argparse
import json

from coact_cps.data.csv_adapter import inspect_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a HAI/SWaT CSV before configuration")
    parser.add_argument("csv_path")
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(inspect_csv(args.csv_path, args.rows), indent=2))


if __name__ == "__main__":
    main()
