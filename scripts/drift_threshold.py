"""Lab 4 Task 5 — derive the drift threshold from this data, not from a tutorial.

    python scripts/drift_threshold.py            # writes reports/lab4-drift-threshold.{json,md}

Two questions, answered by simulation over the real file:

  1. With NO drift, what PSI does a window of N production rows score against the
     training reference? Production rows are the machines the model never trained on —
     the honest stand-in for tomorrow's fleet. The threshold must sit above this, or the
     alert fires on normal variance ("everything alerts constantly").
  2. With drift, how often does that threshold catch it? A threshold above every null
     window that never fires on a real shift is the other failure ("never fires").

The rule. Per window size and PER FEATURE, the threshold is the largest null PSI that
feature scored in NULL_DRAWS windows, rounded UP to the next 0.05 — a per-check false-alarm
rate below 1/NULL_DRAWS for each feature. Per feature because a first pass with one global
threshold showed the floor is set entirely by load_pct: duty cycle is a per-machine trait,
96 unseen machines are few, and its no-drift PSI is 4-6x every other feature's. One global
number therefore blinds temp_c to shifts four times larger than it could otherwise see.

The window is then the smallest N whose thresholds catch the TARGET shift in at least 90%
of windows. The target is +4 °C on temp_c: the smallest whole-degree shift that moves the
production model's mean predicted risk by roughly 10% (measured with --model below).
Smaller shifts exist and are real; they do not change what the model tells anyone.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from monitoring.drift import psi  # noqa: E402
from src import config, data, seeds  # noqa: E402

NULL_DRAWS = 2000
DRIFT_DRAWS = 500
WINDOWS = (250, 500, 1000)
TARGET_SHIFT_C = 4.0
TARGET_POWER = 0.90
OUT = Path("reports/lab4-drift-threshold")


def risk_response(model_dir: Path, pool: pd.DataFrame) -> dict:
    """How much the production model's output moves per degree of temp_c shift."""
    import os

    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow.sklearn

    model = mlflow.sklearn.load_model(str(model_dir))
    x = pool[data.FEATURES]
    base = model.predict_proba(x)[:, 1]
    out = {"baseline_mean_risk": round(float(base.mean()), 4),
           "baseline_flagged_at_0.5": round(float((base > 0.5).mean()), 4)}
    for d in (2, 4, 6, 8):
        p = model.predict_proba(x.assign(temp_c=x.temp_c + d))[:, 1]
        out[f"+{d}C"] = {"mean_risk": round(float(p.mean()), 4),
                         "relative": round(float(p.mean() / base.mean() - 1), 4),
                         "flagged_at_0.5": round(float((p > 0.5).mean()), 4)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("reports/prod-model"),
                    help="the production version's model directory, for the risk response")
    args = ap.parse_args()
    df = data.load_raw(config.REPO_ROOT / "data" / "raw" / "sensors.csv")
    train, val, test = data.split(df, seed=seeds.DEFAULT_SEED)
    reference = {f: train[f].to_numpy(float) for f in data.FEATURES}
    pool = pd.concat([val, test]).reset_index(drop=True)
    rng = np.random.default_rng(0)

    result: dict = {"reference_rows": len(train), "pool_rows": len(pool),
                    "pool_machines": int(pool[data.GROUP].nunique()),
                    "null_draws": NULL_DRAWS, "drift_draws": DRIFT_DRAWS, "windows": {}}
    for n in WINDOWS:
        null = np.empty((NULL_DRAWS, len(data.FEATURES)))
        for i in range(NULL_DRAWS):
            idx = rng.choice(len(pool), size=n, replace=False)
            null[i] = [psi(reference[f], pool[f].to_numpy(float)[idx]) for f in data.FEATURES]
        worst = null.max(axis=1)
        global_threshold = math.ceil(worst.max() / 0.05) * 0.05
        per_feature = {f: round(math.ceil(null[:, j].max() / 0.05) * 0.05, 2)
                       for j, f in enumerate(data.FEATURES)}
        threshold = per_feature["temp_c"]
        limits = np.array([per_feature[f] for f in data.FEATURES])

        def power(transform) -> dict:
            scores = []
            for _ in range(DRIFT_DRAWS):
                idx = rng.choice(len(pool), size=n, replace=False)
                cur = transform(pool.iloc[idx])
                scores.append(psi(reference["temp_c"], cur["temp_c"].to_numpy(float)))
            scores = np.array(scores)
            return {"median_psi": round(float(np.median(scores)), 4),
                    "p05_psi": round(float(np.quantile(scores, 0.05)), 4),
                    "caught_at_threshold": round(float((scores > threshold).mean()), 3),
                    "caught_at_global": round(float((scores > global_threshold).mean()), 3),
                    "caught_at_0.25": round(float((scores > 0.25).mean()), 3)}

        shifts = {f"temp_c +{d:g}C": power(lambda w, d=d: w.assign(temp_c=w.temp_c + d))
                  for d in (2, 4, 6, 8)}
        shifts["temp_c spread x1.5"] = power(
            lambda w: w.assign(temp_c=w.temp_c.mean() + (w.temp_c - w.temp_c.mean()) * 1.5))
        result["windows"][str(n)] = {
            "null_per_feature_p99": {f: round(float(np.quantile(null[:, j], 0.99)), 4)
                                     for j, f in enumerate(data.FEATURES)},
            "null_per_feature_max": {f: round(float(null[:, j].max()), 4)
                                     for j, f in enumerate(data.FEATURES)},
            "null_worst_feature_p50": round(float(np.median(worst)), 4),
            "null_worst_feature_p99": round(float(np.quantile(worst, 0.99)), 4),
            "null_worst_feature_max": round(float(worst.max()), 4),
            "false_alarm_rate_at_0.10": round(float((worst > 0.10).mean()), 4),
            "false_alarm_rate_at_0.25": round(float((worst > 0.25).mean()), 4),
            "global_threshold": round(global_threshold, 2),
            "thresholds": per_feature,
            # Any feature over its own limit, across all null windows: the alert's real
            # false-alarm rate, since the policy ORs the six conditions.
            "false_alarm_rate_per_feature_rule": round(float((null > limits).any(axis=1).mean()), 4),
            "drift": shifts,
        }

    chosen = next((n for n in WINDOWS if result["windows"][str(n)]["drift"]
                   [f"temp_c +{TARGET_SHIFT_C:g}C"]["caught_at_threshold"] >= TARGET_POWER), None)
    result["chosen_window"] = chosen
    result["chosen_thresholds"] = result["windows"][str(chosen)]["thresholds"] if chosen else None
    if args.model.exists():
        result["risk_response"] = risk_response(args.model, pool)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

    lines = ["# Lab 4 Task 5 — the drift threshold, derived", "",
             f"Reference: training split, {result['reference_rows']} rows. Production stand-in: "
             f"{result['pool_rows']} rows from {result['pool_machines']} machines the model never saw. "
             f"{NULL_DRAWS} no-drift windows and {DRIFT_DRAWS} drifted windows per size. "
             "Generated by `scripts/drift_threshold.py`; numbers below are rendered from "
             "`lab4-drift-threshold.json`, not typed.", "",
             "## No drift: what normal looks like", "",
             "Worst feature per window (what a single global threshold has to clear):", "",
             "| window | worst-feature PSI p50 | p99 | max | false alarms at 0.10 | at 0.25 | global threshold | false alarms, per-feature rule |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, w in result["windows"].items():
        lines.append(f"| {n} | {w['null_worst_feature_p50']} | {w['null_worst_feature_p99']} | "
                     f"{w['null_worst_feature_max']} | {w['false_alarm_rate_at_0.10']:.1%} | "
                     f"{w['false_alarm_rate_at_0.25']:.1%} | {w['global_threshold']:.2f} | "
                     f"{w['false_alarm_rate_per_feature_rule']:.2%} |")
    lines += ["", "Per feature: null max -> **threshold** (max rounded up to 0.05):", "",
              "| window | " + " | ".join(data.FEATURES) + " |",
              "|---:|" + "---|" * len(data.FEATURES)]
    for n, w in result["windows"].items():
        lines.append(f"| {n} | " + " | ".join(
            f"{w['null_per_feature_max'][f]} -> **{w['thresholds'][f]:.2f}**" for f in data.FEATURES) + " |")
    lines += ["", "## Drift: how often each threshold catches it (temp_c only)", "",
              "| window | change | median PSI | 5th pct PSI | caught, per-feature | caught, global | caught at 0.25 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for n, w in result["windows"].items():
        for name, s in w["drift"].items():
            lines.append(f"| {n} | {name} | {s['median_psi']} | {s['p05_psi']} | "
                         f"{s['caught_at_threshold']:.1%} | {s['caught_at_global']:.1%} | "
                         f"{s['caught_at_0.25']:.1%} |")
    if "risk_response" in result:
        r = result["risk_response"]
        lines += ["", "## Why +4 °C is the target: what the production model does with it", "",
                  f"Baseline mean predicted risk on unseen machines: {r['baseline_mean_risk']}.", "",
                  "| temp_c shift | mean risk | relative | flagged at 0.5 |", "|---|---:|---:|---:|"]
        for d in (2, 4, 6, 8):
            x = r[f"+{d}C"]
            lines.append(f"| +{d} °C | {x['mean_risk']} | {x['relative']:+.1%} | {x['flagged_at_0.5']:.1%} |")
    lines += ["", f"**Chosen: window {chosen}, thresholds {result['chosen_thresholds']}.**", ""]
    OUT.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
