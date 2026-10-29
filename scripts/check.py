"""Run the same three checks as CI (lint, typecheck, test) locally.

Usage: python scripts/check.py
"""

import subprocess
import sys

CHECKS = {
    "lint": ["uv", "run", "ruff", "check", "src/", "tests/"],
    "typecheck": ["uv", "run", "mypy", "src/", "--ignore-missing-imports"],
    "test": [
        "uv", "run", "pytest", "tests/", "-v",
        "-m", "not slow and not benchmark and not integration",
        "--tb=short", "-x",
    ],
}


def main() -> int:
    failed = []
    for name, cmd in CHECKS.items():
        print(f"\n=== {name}: {' '.join(cmd)}")
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    print("\n" + ("FAILED: " + ", ".join(failed) if failed else "All checks passed"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
