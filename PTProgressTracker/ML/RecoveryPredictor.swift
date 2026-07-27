import Foundation
import CoreML

struct RecoveryPrediction {
    let onTrack: Bool
    let confidence: Double
    let rationale: String
}

/// Computes recovery-pattern features from a patient's logged sessions and
/// runs them through RecoveryGoalClassifier.mlmodel (trained via CreateML on
/// synthetic data, see MLTraining/train_model.swift) to predict whether the
/// patient is on track to meet their recovery goal.
struct RecoveryPredictor {
    private let model: RecoveryGoalClassifier?

    init() {
        model = try? RecoveryGoalClassifier(configuration: MLModelConfiguration())
    }

    /// Requires at least 3 logged sessions to produce a meaningful feature set.
    func predict(from logs: [ExerciseLog], calendar: Calendar = .current, today: Date = Date()) -> RecoveryPrediction? {
        guard logs.count >= 3, let model else { return nil }

        let sorted = logs.sorted { $0.wrappedDate < $1.wrappedDate }
        guard let firstDate = sorted.first?.wrappedDate else { return nil }
        let daysSinceStart = max(1, calendar.dateComponents([.day], from: firstDate, to: today).day ?? 1)

        let streak = StreakCalculator.summary(for: logs, calendar: calendar, today: today)

        let recent7 = sorted.suffix(7)
        let avgPainLast7 = recent7.reduce(0.0) { $0 + Double($1.painLevel) } / Double(max(recent7.count, 1))

        let midpoint = sorted.count / 2
        let earlierHalf = sorted[0..<max(midpoint, 1)]
        let laterHalf = sorted[midpoint...]
        let earlierAvgPain = earlierHalf.reduce(0.0) { $0 + Double($1.painLevel) } / Double(max(earlierHalf.count, 1))
        let laterAvgPain = laterHalf.reduce(0.0) { $0 + Double($1.painLevel) } / Double(max(laterHalf.count, 1))
        let painTrend = laterAvgPain - earlierAvgPain

        let completionRatios = sorted.map { log -> Double in
            guard let target = ExerciseLibrary.byID(log.wrappedExerciseID)?.targetReps, target > 0 else { return 1.0 }
            return Double(log.reps) / Double(target)
        }
        let completionRatio = completionRatios.reduce(0, +) / Double(max(completionRatios.count, 1))

        do {
            let input = RecoveryGoalClassifierInput(
                daysSinceStart: Int64(daysSinceStart),
                adherenceRatio: streak.adherenceLast7Days,
                avgPainLast7: avgPainLast7,
                painTrend: painTrend,
                completionRatio: completionRatio,
                currentStreak: Int64(streak.currentStreak)
            )
            let output = try model.prediction(input: input)
            let confidence = output.onTrackProbability[output.onTrack] ?? 0.5
            let onTrack = output.onTrack == 1

            return RecoveryPrediction(
                onTrack: onTrack,
                confidence: confidence,
                rationale: rationale(
                    onTrack: onTrack,
                    adherence: streak.adherenceLast7Days,
                    painTrend: painTrend,
                    completionRatio: completionRatio
                )
            )
        } catch {
            return nil
        }
    }

    private func rationale(onTrack: Bool, adherence: Double, painTrend: Double, completionRatio: Double) -> String {
        if onTrack {
            if painTrend < 0 {
                return "Pain is trending down and adherence is strong."
            }
            return "Adherence and exercise completion are consistent."
        } else {
            if adherence < 0.5 {
                return "Adherence has dropped — try logging sessions more consistently."
            }
            if painTrend > 0 {
                return "Pain levels are trending up — consider discussing with your therapist."
            }
            return "Completion ratio is below target — aim for full sets and reps."
        }
    }
}
