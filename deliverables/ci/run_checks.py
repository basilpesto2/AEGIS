from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent


def main() -> None:
    commands = [
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(ROOT / "reproducible_pipeline" / "tests"),
            "-v",
        ],
        [sys.executable, str(ROOT / "verify_deliverables.py")],
    ]
    for command in commands:
        print(f"+ {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)
    print("AEGIS deliverables CI: PASS")


if __name__ == "__main__":
    main()
