"""Lab 4 — a production input stream, for the dashboard and the drift detector.

    python scripts/replay_traffic.py --source data/raw/sensors.csv --rate 4 --minutes 20
    python scripts/replay_traffic.py --inject data/current.csv --inject-after-min 15 ...

One process for the whole exercise, switching source mid-stream: a restart would empty the
window, and a detector should see drifted rows displace clean ones the way production
would, not a clean window replaced wholesale.

Sends rows to the running service one /predict at a time, and every --flush-s seconds
ships the most recent WINDOW inputs it sent to object storage as lab4/inputs/latest.csv —
standing in for the request logging a managed endpoint would do (the endpoint is off;
Lab 3 spent the budget). The scheduled detector reads that object.

Rows come from the machines the model did NOT train on, shuffled across machines: a
real fleet reports concurrently, and a window holding twenty machines in a row is a
fleet-mix change, not a steady state. Measured: the first 500 rows of the test split in
file order score load_pct PSI 0.33 with no drift at all.

--bad-rate sends that fraction of requests with an out-of-range value, so the 4xx panel
has something to show; rejected rows never enter the window (the service and this script
agree on that).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from monitoring.drift import WINDOW  # noqa: E402
from src import config, data, seeds  # noqa: E402

LOG = Path("reports/lab4-replay.jsonl")


def production_rows(source: Path, seed: int) -> pd.DataFrame:
    df = pd.read_csv(source)
    train, _, _ = data.split(pd.read_csv(config.REPO_ROOT / "data" / "raw" / "sensors.csv"),
                             seed=seeds.DEFAULT_SEED)
    unseen = df[~df[data.GROUP].isin(set(train[data.GROUP]))]
    return unseen.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--target", default="http://localhost:8080")
    ap.add_argument("--rate", type=float, default=4.0, help="requests per second")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--flush-s", type=float, default=60.0)
    ap.add_argument("--bad-rate", type=float, default=0.01)
    ap.add_argument("--inject", type=Path, help="switch to this source mid-stream")
    ap.add_argument("--inject-after-min", type=float, default=10.0)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = production_rows(args.source, args.seed)
    label = "clean"
    inject_at = (time.monotonic() + args.inject_after_min * 60) if args.inject else None
    rng = np.random.default_rng(args.seed)
    window: deque[dict] = deque(maxlen=WINDOW)
    adapter = None
    if not args.no_upload:
        from cloudlayer.factory import get_adapter
        adapter = get_adapter(config.load())

    session = requests.Session()
    period = 1.0 / args.rate
    deadline = time.monotonic() + args.minutes * 60
    next_flush = time.monotonic() + args.flush_s
    sent = ok = rejected = failed = 0
    LOG.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        if not window:
            return
        frame = pd.DataFrame(list(window))
        record = {"event": "flush", "label": label, "rows": len(frame),
                  "newest": frame["sent_at"].max(), "at": _now(),
                  "temp_c_mean": round(float(frame["temp_c"].mean()), 3),
                  "sent": sent, "ok": ok, "rejected": rejected, "failed": failed}
        if adapter is not None:
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "latest.csv"
                frame.to_csv(local, index=False)
                record["uri"] = adapter.upload(str(local), "lab4/inputs/latest.csv")
        with LOG.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    with LOG.open("a") as fh:
        fh.write(json.dumps({"event": "start", "source": str(args.source),
                             "inject": str(args.inject), "inject_after_min": args.inject_after_min,
                             "rate": args.rate, "at": _now()}) + "\n")
    i = 0
    while time.monotonic() < deadline:
        started = time.monotonic()
        if inject_at is not None and started >= inject_at:
            rows, label, inject_at, i = production_rows(args.inject, args.seed), "injected", None, 0
            event = {"event": "inject", "source": str(args.inject), "at": _now()}
            with LOG.open("a") as fh:
                fh.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
        row = {f: float(rows.iloc[i % len(rows)][f]) for f in data.FEATURES}
        i += 1
        bad = rng.random() < args.bad_rate
        payload = {**row, "load_pct": 250.0} if bad else row
        try:
            r = session.post(f"{args.target}/predict", json=payload, timeout=10)
            sent += 1
            if r.status_code == 200:
                ok += 1
                window.append({**row, "sent_at": _now()})
            elif 400 <= r.status_code < 500:
                rejected += 1
            else:
                failed += 1
        except requests.RequestException:
            failed += 1
        if time.monotonic() >= next_flush:
            flush()
            next_flush += args.flush_s
        time.sleep(max(0.0, period - (time.monotonic() - started)))
    flush()
    return 0


def _now() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int(t % 1 * 1000):03d}Z"


if __name__ == "__main__":
    raise SystemExit(main())
