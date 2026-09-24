"""Feature contract shared with the iOS app.

A line-for-line port of the feature computation in
PTProgressTracker/ML/RecoveryPredictor.swift and
PTProgressTracker/ViewModels/StreakCalculator.swift. The simulator emits raw
exercise logs and every training row is computed from those logs here, so the
model only ever sees feature combinations the app can actually produce.

PTProgressTrackerTests/Fixtures/feature_parity_cases.json is written from this
module by make_parity_fixture.py; FeatureParityTests.swift asserts the Swift
implementation reproduces it.

Semantics both sides must agree on:
  * Days are calendar days in UTC (the app uses the device calendar; the parity
    fixture pins both implementations to UTC).
  * Timestamps are seconds since the Unix epoch.
  * Logs are ordered by timestamp. Swift's sort does not define an order for
    ties, so the simulator never emits two logs with the same timestamp.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

SECONDS_PER_DAY = 86_400

# RecoveryPredictor.predict returns nil below this many logs.
MIN_LOGS = 3

# Mirrors ExerciseLibrary.all in PTProgressTracker/Models/Exercise.swift.
TARGET_REPS = {
    "knee-extension": 12,
    "shoulder-flexion": 10,
    "pelvic-tilt": 10,
    "ankle-circles": 15,
    "glute-bridge": 12,
    "wall-slide": 10,
    "heel-slide": 12,
    "clamshell": 15,
}

FEATURE_NAMES = (
    "daysSinceStart",
    "adherenceRatio",
    "avgPainLast7",
    "painTrend",
    "completionRatio",
    "currentStreak",
)


@dataclass(frozen=True)
class Log:
    timestamp: float  # seconds since epoch, UTC
    exercise_id: str
    reps: int
    pain: int


@dataclass(frozen=True)
class StreakSummary:
    current_streak: int
    longest_streak: int
    adherence_last_7_days: float


def utc_day(timestamp: float) -> int:
    """Calendar-day index (days since epoch) of a UTC timestamp."""
    return math.floor(timestamp / SECONDS_PER_DAY)


def streak_summary(logs: Sequence[Log], today: float) -> StreakSummary:
    """StreakCalculator.summary(for:calendar:today:)."""
    logged_days = {utc_day(log.timestamp) for log in logs}
    today_day = utc_day(today)

    # A streak survives until the end of today, so start from yesterday if
    # nothing has been logged yet today.
    cursor = today_day if today_day in logged_days else today_day - 1
    current = 0
    while cursor in logged_days:
        current += 1
        cursor -= 1

    longest = 0
    running = 0
    previous: Optional[int] = None
    for day in sorted(logged_days):
        running = running + 1 if previous is not None and previous + 1 == day else 1
        longest = max(longest, running)
        previous = day

    last7 = sum(1 for offset in range(7) if today_day - offset in logged_days)
    return StreakSummary(current, longest, last7 / 7.0)


def _mean(values: Sequence[float]) -> float:
    # Swift: reduce(0.0, +) / Double(max(count, 1)), summed left to right.
    total = 0.0
    for value in values:
        total += value
    return total / max(len(values), 1)


def compute_features(logs: Sequence[Log], today: float) -> Optional[dict]:
    """RecoveryPredictor feature vector, or None when the app shows no prediction."""
    if len(logs) < MIN_LOGS:
        return None

    ordered = sorted(logs, key=lambda log: log.timestamp)
    first = ordered[0].timestamp
    # Calendar.dateComponents([.day], from:to:) counts whole elapsed days.
    days_since_start = max(1, math.floor((today - first) / SECONDS_PER_DAY))

    streak = streak_summary(logs, today)

    avg_pain_last7 = _mean([float(log.pain) for log in ordered[-7:]])

    midpoint = len(ordered) // 2
    earlier = ordered[0 : max(midpoint, 1)]
    later = ordered[midpoint:]
    pain_trend = _mean([float(log.pain) for log in later]) - _mean([float(log.pain) for log in earlier])

    def completion(log: Log) -> float:
        target = TARGET_REPS.get(log.exercise_id)
        if target is None or target <= 0:
            return 1.0
        return float(log.reps) / float(target)

    completion_ratio = _mean([completion(log) for log in ordered])

    return {
        "daysSinceStart": days_since_start,
        "adherenceRatio": streak.adherence_last_7_days,
        "avgPainLast7": avg_pain_last7,
        "painTrend": pain_trend,
        "completionRatio": completion_ratio,
        "currentStreak": streak.current_streak,
    }
