"""Lab 4 — unit tests for the drift statistics.

Fast, no network, no data file. Each one pins a property the alerting depends on: if PSI
of a sample against itself were not ~0, or a shifted sample did not score higher than an
unshifted one, every threshold in the README would be a number about a broken function.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring import drift

FEATURES = ["a", "b"]


@pytest.fixture(scope="module")
def reference():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(70, 9, 5000), "b": rng.uniform(0, 100, 5000)})


def test_psi_of_a_sample_against_itself_is_near_zero(reference):
    values = reference["a"].to_numpy()
    assert drift.psi(values, values) < 1e-6


def test_psi_grows_with_the_size_of_a_mean_shift(reference):
    values = reference["a"].to_numpy()
    scores = [drift.psi(values, values + shift) for shift in (0, 2, 6, 12)]
    assert scores == sorted(scores), f"PSI not monotonic in the shift: {scores}"
    assert scores[-1] > drift.PSI_MODERATE


def test_psi_sees_a_change_in_spread_the_mean_does_not(reference):
    """The scale mode of inject_drift: the mean stays put and PSI must still move."""
    values = reference["a"].to_numpy()
    widened = values.mean() + (values - values.mean()) * 2.0
    assert abs(widened.mean() - values.mean()) < 1e-9
    assert drift.psi(values, widened) > drift.PSI_MODERATE


def test_psi_survives_empty_bins(reference):
    """Every current value in one bin leaves nine empty. Without smoothing that is log(0)."""
    values = reference["a"].to_numpy()
    score = drift.psi(values, np.full(500, values.max() + 50))
    assert np.isfinite(score) and score > 1


def test_psi_of_a_constant_reference_is_zero_not_a_crash():
    assert drift.psi(np.ones(100), np.ones(100) * 2) == 0.0


def test_ks_bounds_and_identity(reference):
    values = reference["a"].to_numpy()
    assert drift.ks_statistic(values, values) == 0.0
    assert drift.ks_statistic(values, values + 1000) == 1.0


def test_profile_scoring_matches_raw_scoring(reference):
    """The service scores against the stored profile, the scheduled job against raw data.
    If the two disagreed, the dashboard and the alert would be measuring different things."""
    profile = drift.build_profile(reference, FEATURES)
    rng = np.random.default_rng(1)
    current = rng.normal(73, 9, 500)
    raw = drift.psi(reference["a"].to_numpy(), current)
    stored = drift.psi_from_profile(profile["features"]["a"], current, profile["rows"])
    assert stored == pytest.approx(raw, abs=1e-4)


def test_compare_sorts_worst_first_and_labels_verdicts(reference):
    current = reference.copy()
    current["b"] = current["b"] * 0.01          # a unit change: percent became a fraction
    results = drift.compare(reference, current, FEATURES)
    assert results[0].feature == "b"
    assert results[0].verdict == "significant"
    assert results[1].verdict == "stable"


@pytest.mark.parametrize("score,expected", [
    (0.0, "stable"), (drift.PSI_NO_CHANGE, "moderate"), (drift.PSI_MODERATE, "significant"),
])
def test_verdict_boundaries_are_inclusive_upward(score, expected):
    assert drift.verdict_for(score) == expected


def test_every_feature_has_a_derived_alert_limit():
    """A feature added to the model without a limit would fall back to the tutorial 0.25,
    which this data shows catches a +4 °C shift 3.6% of the time."""
    from src import data

    assert set(drift.ALERT_PSI) == set(data.FEATURES)
    assert all(0 < v < drift.PSI_MODERATE * 2 for v in drift.ALERT_PSI.values())
