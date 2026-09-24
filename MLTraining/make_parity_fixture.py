"""Write the feature-parity fixture shared by the Python and Swift test suites.

    python3 make_parity_fixture.py

Each case is a log history, a "today", and the six features features.py
computes for it (or null when the app would show no prediction). The Swift
test (PTProgressTrackerTests/FeatureParityTests.swift) feeds the same histories
through RecoveryFeatures and must match to 1e-9. Rerun this whenever
features.py changes, and make the same change in Swift.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import features as F
import simulate as S

OUT = Path(__file__).resolve().parent.parent / "PTProgressTrackerTests" / "Fixtures" / "feature_parity_cases.json"
T0 = S.EPOCH_2024
DAY = F.SECONDS_PER_DAY


def stamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def at(day: int, hour: int = 9, minute: int = 0) -> int:
    return T0 + day * DAY + hour * 3600 + minute * 60


def end_of(day: int) -> int:
    return T0 + (day + 1) * DAY - 1


def case(name: str, logs: list, today: int) -> dict:
    values = F.compute_features(logs, float(today))
    return {
        "name": name,
        "today": stamp(today),
        "logs": [
            {"timestamp": stamp(l.timestamp), "exerciseID": l.exercise_id, "reps": l.reps, "painLevel": l.pain}
            for l in logs
        ],
        "expected": values,
    }


def L(ts, pain, reps=12, exercise="knee-extension"):
    return F.Log(float(ts), exercise, reps, pain)


def hand_written() -> list:
    return [
        case("two logs: no prediction", [L(at(0), 5), L(at(1), 4)], end_of(1)),
        case("three logs on consecutive days", [L(at(0), 6), L(at(1), 4), L(at(2), 2)], end_of(2)),
        case("streak survives until the end of today", [L(at(d), 5) for d in range(4)], at(4, 20)),
        case("a missed day breaks the streak", [L(at(d), 5) for d in (0, 1, 3, 4)], end_of(4)),
        case(
            "several logs a day count once",
            [L(at(0, 8), 6), L(at(0, 8, 3), 5, exercise="heel-slide"), L(at(1, 19), 5), L(at(1, 19, 3), 4, exercise="heel-slide")],
            end_of(1),
        ),
        case(
            "average pain uses the last seven logs, not days",
            [L(at(d), 9) for d in range(3)] + [L(at(3, 8 + i), p) for i, p in enumerate([1, 2, 3, 4, 5, 6, 7])],
            end_of(3),
        ),
        case("pain trend with an odd number of logs", [L(at(d), p) for d, p in enumerate([8, 7, 5, 4, 3])], end_of(4)),
        case(
            "unknown exercise counts as full completion",
            [L(at(0), 5, reps=6), L(at(1), 5, reps=15, exercise="clamshell"), L(at(2), 5, reps=3, exercise="not-in-library")],
            end_of(2),
        ),
        case("days since start counts whole days", [L(at(0, 18), 5), L(at(1), 5), L(at(2), 5)], at(6, 17, 59)),
        case("first day never reports zero days", [L(at(0, h), 5) for h in (8, 9, 10)], at(0, 11)),
        case("adherence window is today and the six days before", [L(at(d), 5) for d in (0, 3, 9, 10, 15)], end_of(15)),
        case("reps above target", [L(at(d), 3, reps=20, exercise="pelvic-tilt") for d in range(3)], end_of(2)),
    ]


def simulated(n_cases: int = 20) -> list:
    cohort = S.simulate(S.SimConfig(n_patients=300), seed=42)
    rng = np.random.default_rng(0)
    cases = []
    for patient in rng.choice(len(cohort.patients), size=n_cases, replace=False):
        week = int(rng.integers(1, 6))
        enroll = int(cohort.patients.enroll_day[patient])
        today = S.snapshot_time(enroll, week)
        history = cohort.logs[(cohort.logs.patient_id == patient) & (cohort.logs.timestamp <= today)]
        logs = [L(r.timestamp, int(r.pain), int(r.reps), r.exercise_id) for r in history.itertuples()]
        cases.append(case(f"simulated patient {patient}, week {week}", logs, today))
    return cases


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generatedBy": "MLTraining/make_parity_fixture.py",
        "calendar": "gregorian, UTC",
        "cases": hand_written() + simulated(),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"Wrote {len(payload['cases'])} cases to {OUT}")


if __name__ == "__main__":
    main()
