"""Lab 4 Task 5 — the alert that turns a drift score into an email.

    python scripts/drift_alerting.py --email you@example.com

Idempotent: re-running updates the one policy in place, so changing a limit in
monitoring/drift.ALERT_PSI and re-running is how the alert follows the code. The limits
come from that dict and nowhere else — a threshold typed into a console is a threshold
nobody can find in review. Labelled course=itcs355, lab=4, so `make teardown LAB=4`
deletes the policy and its channel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter  # noqa: E402
from monitoring.drift import ALERT_PSI  # noqa: E402
from src import config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--record", type=Path, default=Path("reports/lab4-alerting.json"))
    args = ap.parse_args()

    cfg = config.load()
    created = get_adapter(cfg).ensure_metric_alert("drift_psi", ALERT_PSI, args.email, cfg.tags(4))
    record = {**created, "thresholds": ALERT_PSI, "metric": "drift_psi", "tags": cfg.tags(4)}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
