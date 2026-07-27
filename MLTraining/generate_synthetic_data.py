#!/usr/bin/env python3
"""
Generates synthetic physical-therapy adherence/recovery data for training
RecoveryGoalClassifier.mlmodel via CreateML (see train_model.swift).

Each row represents a snapshot of a patient's logging history at some point
in their plan, reduced to the same features RecoveryPredictor.swift computes
at runtime from Core Data logs:

  daysSinceStart        - days elapsed since the recovery plan started
  adherenceRatio         - fraction of days in the last 7 with a logged session (0-1)
  avgPainLast7           - average self-reported pain level over the last 7 sessions (0-10)
  painTrend              - avg pain in the most recent half minus the earlier half of
                            the session history (negative = improving)
  completionRatio        - average (reps completed / target reps) across sessions
  currentStreak          - consecutive days logged up to today

Label:
  onTrack (0/1) - whether the patient is on track to meet their recovery goal.

The generative rule is a noisy logistic combination of the features that
reflects clinical intuition (higher adherence + improving pain + good
completion + streak => more likely on track), not real patient data.
"""
import csv
import random
import math

random.seed(42)

N_ROWS = 4000
OUTPUT_PATH = "synthetic_recovery_data.csv"


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def generate_row():
    days_since_start = random.randint(1, 120)
    adherence = min(1.0, max(0.0, random.betavariate(2.2, 1.6)))
    avg_pain = min(10.0, max(0.0, random.gauss(4.5, 2.2)))
    # painTrend: improving patients trend negative; scale with adherence noise
    pain_trend = random.gauss(-1.0 * adherence + 0.5, 1.4)
    completion_ratio = min(1.3, max(0.0, random.gauss(0.55 + 0.4 * adherence, 0.22)))
    current_streak = max(0, int(random.gauss(adherence * 10, 4)))

    # Clinical-intuition score: higher adherence, lower/negative pain trend,
    # higher completion, longer streak => higher probability of being on track.
    score = (
        2.6 * adherence
        + 1.4 * completion_ratio
        - 0.55 * pain_trend
        + 0.12 * current_streak
        - 0.015 * days_since_start
        - 1.1
    )
    score += random.gauss(0, 0.6)  # label noise
    prob_on_track = sigmoid(score)
    on_track = 1 if random.random() < prob_on_track else 0

    return {
        "daysSinceStart": days_since_start,
        "adherenceRatio": round(adherence, 4),
        "avgPainLast7": round(avg_pain, 2),
        "painTrend": round(pain_trend, 3),
        "completionRatio": round(completion_ratio, 3),
        "currentStreak": current_streak,
        "onTrack": on_track,
    }


def main():
    fieldnames = [
        "daysSinceStart",
        "adherenceRatio",
        "avgPainLast7",
        "painTrend",
        "completionRatio",
        "currentStreak",
        "onTrack",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(N_ROWS):
            writer.writerow(generate_row())

    print(f"Wrote {N_ROWS} synthetic rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
