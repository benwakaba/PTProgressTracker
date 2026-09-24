import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import features as F

FIXTURE = Path(__file__).resolve().parents[2] / "PTProgressTrackerTests" / "Fixtures" / "feature_parity_cases.json"
DAY = F.SECONDS_PER_DAY
T0 = 1_704_067_200  # 2024-01-01T00:00:00Z


def at(day: int, hour: float = 9) -> float:
    return T0 + day * DAY + hour * 3600


def log(day, pain, reps=12, exercise="knee-extension", hour=9):
    return F.Log(at(day, hour), exercise, reps, pain)


def end_of(day: int) -> float:
    return at(day + 1, 0) - 1


def test_fewer_than_three_logs_gives_no_prediction():
    assert F.compute_features([log(0, 5), log(1, 5)], end_of(1)) is None


def test_three_logs_split_one_earlier_two_later():
    # midpoint = 3 // 2 = 1: earlier = [first], later = [second, third]
    values = F.compute_features([log(0, 6), log(1, 4), log(2, 2)], end_of(2))
    assert values["painTrend"] == pytest.approx((4 + 2) / 2 - 6)
    assert values["avgPainLast7"] == pytest.approx(4)
    assert values["currentStreak"] == 3
    assert values["adherenceRatio"] == pytest.approx(3 / 7)


def test_streak_survives_until_end_of_today():
    logs = [log(0, 5), log(1, 5), log(2, 5)]
    # Nothing logged on day 3 yet: the streak still counts back from day 2.
    assert F.streak_summary(logs, at(3, 20)).current_streak == 3
    # A full day missed breaks it.
    assert F.streak_summary(logs, at(4, 20)).current_streak == 0


def test_gap_breaks_streak_and_several_logs_a_day_count_once():
    logs = [log(0, 5), log(0, 5, hour=10), log(2, 5), log(3, 5), log(3, 4, hour=18)]
    summary = F.streak_summary(logs, end_of(3))
    assert summary.current_streak == 2
    assert summary.longest_streak == 2
    assert summary.adherence_last_7_days == pytest.approx(3 / 7)


def test_adherence_window_is_today_and_six_days_before():
    logs = [log(d, 5) for d in (0, 3, 9, 10, 15)]
    # Window for day 15 is days 9..15.
    assert F.streak_summary(logs, end_of(15)).adherence_last_7_days == pytest.approx(3 / 7)


def test_avg_pain_uses_the_last_seven_logs_not_days():
    logs = [log(d, 9) for d in range(3)] + [log(3, p, hour=8 + i) for i, p in enumerate([1, 2, 3, 4, 5, 6, 7])]
    assert F.compute_features(logs, end_of(3))["avgPainLast7"] == pytest.approx(4)


def test_pain_trend_with_odd_count():
    pains = [8, 7, 5, 4, 3]  # midpoint 2: earlier [8, 7], later [5, 4, 3]
    values = F.compute_features([log(d, p) for d, p in enumerate(pains)], end_of(4))
    assert values["painTrend"] == pytest.approx(4 - 7.5)


def test_completion_ratio_and_unknown_exercise():
    logs = [log(0, 5, reps=6), log(1, 5, reps=12), log(2, 5, reps=3, exercise="not-in-library")]
    # 6/12, 12/12, and 1.0 for an exercise the library does not know.
    assert F.compute_features(logs, end_of(2))["completionRatio"] == pytest.approx((0.5 + 1 + 1) / 3)


def test_days_since_start_counts_whole_elapsed_days():
    logs = [log(0, 5, hour=18), log(1, 5), log(2, 5)]
    assert F.compute_features(logs, at(6, 17.99))["daysSinceStart"] == 5
    assert F.compute_features(logs, at(6, 18.01))["daysSinceStart"] == 6
    # Never below 1, even on the first day.
    assert F.compute_features([log(0, 5, hour=h) for h in (8, 9, 10)], at(0, 11))["daysSinceStart"] == 1


def test_order_of_input_does_not_matter():
    logs = [log(d, p) for d, p in enumerate([6, 5, 5, 3, 2])]
    assert F.compute_features(logs, end_of(4)) == F.compute_features(list(reversed(logs)), end_of(4))


def _parse(stamp: str) -> float:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def test_parity_fixture_matches_this_implementation():
    """The Swift test asserts the app reproduces this fixture; this one keeps it current.

    If it fails, features.py changed: regenerate with make_parity_fixture.py and
    make the same change in RecoveryFeatures (Swift).
    """
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) >= 20
    for case in cases:
        logs = [F.Log(_parse(l["timestamp"]), l["exerciseID"], l["reps"], l["painLevel"]) for l in case["logs"]]
        values = F.compute_features(logs, _parse(case["today"]))
        if case["expected"] is None:
            assert values is None, case["name"]
        else:
            assert values is not None, case["name"]
            for name in F.FEATURE_NAMES:
                assert values[name] == pytest.approx(case["expected"][name], abs=1e-12), (case["name"], name)
