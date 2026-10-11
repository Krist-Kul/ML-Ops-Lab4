"""Lab 4 — drift detection.

PSI and Kolmogorov–Smirnov, implemented directly so you can see what the numbers mean.
Evidently is allowed instead, but you must still be able to explain what your chosen
statistic measures and why your threshold is what it is.

    python -m monitoring.drift --reference data/raw/sensors.csv --current data/current.csv

A threshold copied from a tutorial is not a justified threshold, and Lab 4 grades the
justification, not the code.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Conventional PSI reading, and it IS only conventional — it comes from credit scoring,
# where features are stable and volumes are large. Your problem may warrant something
# tighter or looser. TODO(Lab 4): state your threshold and why, in your README.
PSI_NO_CHANGE = 0.10
PSI_MODERATE = 0.25

# The scheduled detector's settings. Lab 4 Task 5 commits these, with the measurement
# behind them, before any drift is injected.
WINDOW = 500
ALERT_PSI = PSI_MODERATE


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ref_mean: float
    cur_mean: float
    verdict: str


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index.

    Bins the reference into quantiles and compares the proportion of mass falling in each.
    Sensitive to changes in shape, not only in mean — which is why a feature can drift
    badly while its average looks untouched.
    """
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    # Laplace smoothing: an empty bin would otherwise make the log term infinite.
    ref_prop = (ref_counts + 1) / (ref_counts.sum() + len(ref_counts))
    cur_prop = (cur_counts + 1) / (cur_counts.sum() + len(cur_counts))

    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def build_profile(reference: pd.DataFrame, features: list[str], bins: int = 10) -> dict:
    """The reference window reduced to what a scorer needs: decile edges and the share of
    reference mass in each bin, plus mean and std for the contract test.

    The service scores its rolling window against this, so it never needs the training
    data — only these few numbers per feature, which ship inside the image.
    """
    profile: dict = {"bins": bins, "rows": int(len(reference)), "features": {}}
    for feature in features:
        values = reference[feature].to_numpy(dtype=float)
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1))[1:-1])
        counts = np.bincount(np.searchsorted(edges, values, side="right"),
                             minlength=len(edges) + 1)
        profile["features"][feature] = {
            "edges": [round(float(e), 6) for e in edges],
            "proportions": [round(float(c) / len(values), 6) for c in counts],
            "mean": round(float(values.mean()), 6),
            "std": round(float(values.std()), 6),
        }
    return profile


def psi_from_profile(feature_profile: dict, current: np.ndarray, rows: int) -> float:
    """PSI of `current` against a stored profile. Same smoothing as psi(), so the two agree
    to within the rounding of the stored proportions."""
    edges = np.asarray(feature_profile["edges"], dtype=float)
    n_bins = len(edges) + 1
    if n_bins < 3 or len(current) == 0:
        return 0.0
    # Back to counts, so the Laplace smoothing is the same +1 per bin that psi() applies.
    ref_counts = np.asarray(feature_profile["proportions"], dtype=float) * rows
    cur_counts = np.bincount(np.searchsorted(edges, current, side="right"), minlength=n_bins)
    ref_prop = (ref_counts + 1) / (ref_counts.sum() + n_bins)
    cur_prop = (cur_counts + 1) / (cur_counts.sum() + n_bins)
    return float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic: the largest gap between the two CDFs.

    Complements PSI. KS is more sensitive to a shift in location; PSI to a change in shape.
    Reporting both, and noticing when they disagree, is worth more than either alone.
    """
    ref = np.sort(reference)
    cur = np.sort(current)
    pooled = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, pooled, side="right") / len(ref)
    cdf_cur = np.searchsorted(cur, pooled, side="right") / len(cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def verdict_for(score: float) -> str:
    if score < PSI_NO_CHANGE:
        return "stable"
    if score < PSI_MODERATE:
        return "moderate"
    return "significant"


def compare(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> list[FeatureDrift]:
    results = []
    for feature in features:
        ref = reference[feature].to_numpy(dtype=float)
        cur = current[feature].to_numpy(dtype=float)
        score = psi(ref, cur)
        results.append(FeatureDrift(
            feature=feature,
            psi=round(score, 5),
            ks_statistic=round(ks_statistic(ref, cur), 5),
            ref_mean=round(float(ref.mean()), 4),
            cur_mean=round(float(cur.mean()), 4),
            verdict=verdict_for(score),
        ))
    return sorted(results, key=lambda r: r.psi, reverse=True)


def _read_current(source: str) -> pd.DataFrame:
    """A local path, or a storage URI fetched through the adapter. drift.py never parses
    the URI itself: which scheme means which provider is cloudlayer's business."""
    if "://" not in source:
        return pd.read_csv(source)
    import tempfile

    from cloudlayer.factory import get_adapter
    from src import config

    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "current.csv"
        get_adapter(config.load(strict=False)).download(source, str(local))
        return pd.read_csv(local)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import config, data, seeds

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--current", help="CSV path or storage URI of recent production inputs")
    ap.add_argument("--window", type=int, default=WINDOW,
                    help="score the most recent N rows of --current")
    ap.add_argument("--threshold", type=float, default=ALERT_PSI,
                    help="alert above this PSI. Justified in the README, not a default.")
    ap.add_argument("--out", type=Path, default=Path("reports/drift.json"))
    ap.add_argument("--emit", action="store_true", help="send scores as cloud metrics")
    ap.add_argument("--write-profile", type=Path, metavar="PATH",
                    help="write the reference profile (training split) and exit")
    args = ap.parse_args()

    reference = pd.read_csv(args.reference)
    if args.reference == Path("data/raw/sensors.csv"):
        # The reference window is what the model was trained on, not the whole file: the
        # validation and test machines are exactly the "new machines" production sees.
        reference, _, _ = data.split(reference, seed=seeds.DEFAULT_SEED)

    if args.write_profile:
        args.write_profile.write_text(json.dumps(build_profile(reference, data.FEATURES), indent=2) + "\n")
        print(f"wrote {args.write_profile}  ({len(reference)} reference rows)")
        return 0
    if not args.current:
        ap.error("--current is required unless --write-profile is given")

    current = _read_current(args.current)
    if "sent_at" in current.columns:
        current = current.sort_values("sent_at")
    current = current.tail(args.window)
    if len(current) < args.window:
        # A short window is not "no drift"; it is an unknown. The null distribution behind
        # ALERT_PSI was measured at WINDOW rows, and PSI on fewer rows is noisier.
        print(f"WARN  only {len(current)} rows, threshold was set for {args.window}")
    results = compare(reference, current, data.FEATURES)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": args.current, "rows": len(current), "threshold": args.threshold,
        "newest_input": str(current["sent_at"].max()) if "sent_at" in current else None,
        "features": [asdict(r) for r in results],
    }, indent=2))

    print(f"{'feature':<22}{'psi':>10}{'ks':>10}  verdict")
    for r in results:
        print(f"{r.feature:<22}{r.psi:>10.5f}{r.ks_statistic:>10.5f}  {r.verdict}")

    if args.emit:
        from cloudlayer.factory import get_adapter
        adapter = get_adapter(config.load(strict=False))
        for r in results:
            adapter.emit_metric("drift_psi", r.psi, labels={"feature": r.feature})

    breached = [r for r in results if r.psi >= args.threshold]
    if breached:
        print(f"\nALERT  {len(breached)} feature(s) above threshold {args.threshold}: "
              + ", ".join(r.feature for r in breached))
        print("Before you retrain: is this drift, or is it a broken upstream pipeline? "
              "Retraining on corrupted data destroys a working model faster than any "
              "schedule would.")
        return 2
    print(f"\nOK  no feature above threshold {args.threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
