from __future__ import annotations

import argparse
import dataclasses
import json
import sys

from src.monitoring.drift import DriftDetector

_EXIT_CODES: dict[str, int] = {"stable": 0, "warning": 1, "critical": 2}


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight drift detection CLI")
    parser.add_argument(
        "--reference",
        type=str,
        help="Path to saved DriftDetector joblib file",
    )
    parser.add_argument(
        "--current",
        type=str,
        help="Path to JSON file containing {\"texts\": [...], \"labels\": [...]}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write DriftReport JSON (default: stdout only)",
    )

    args = parser.parse_args()

    if args.reference is None or args.current is None:
        parser.print_usage()
        sys.exit(0)

    detector = DriftDetector.load(args.reference)

    with open(args.current) as f:
        data = json.load(f)

    report = detector.detect(data["texts"], data["labels"])
    report_json = json.dumps(dataclasses.asdict(report), indent=2)

    print(report_json)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report_json)

    sys.exit(_EXIT_CODES[report.status])


if __name__ == "__main__":
    main()
