#!/usr/bin/env python3
"""
Generate calibration chart for V4 model (thin wrapper).

This script delegates to the unified calibration chart generator with --version v4.
Use it for backward compatibility, or run generate_calibration_chart.py --version v4 directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run the unified script with --version v4 (strip any existing --version from argv)
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def main() -> None:
    argv = sys.argv[1:]
    new_args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--version" and i + 1 < len(argv):
            i += 2
            continue
        new_args.append(argv[i])
        i += 1
    sys.argv = [sys.argv[0], "--version", "v4"] + new_args
    from production.scripts.generate_calibration_chart import main as unified_main
    unified_main()


if __name__ == "__main__":
    main()
