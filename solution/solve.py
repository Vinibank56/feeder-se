#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from identify import data_root, recover

OUT = Path("/app/feeder_se.json")
LOCAL = Path(__file__).resolve().parents[1] / "feeder_se.local.json"


def main() -> None:
    proto = recover(data_root())
    text = json.dumps(proto, indent=2) + "\n"
    try:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text, encoding="utf-8")
    except OSError:
        LOCAL.write_text(text, encoding="utf-8")
        return
    if not OUT.exists():
        LOCAL.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
