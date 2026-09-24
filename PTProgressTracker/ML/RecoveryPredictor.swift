import Foundation
import CoreML

/// The six model inputs, computed exactly as MLTraining/features.py computes
/// them for every training row. FeatureParityTests holds both implementations
/// to the same fixture, so the model never sees inputs it wasn't trained on.
struct RecoveryFeatures: Equatable {
    static let minimumLogs = 3

    let daysSinceStart: Int
    let adherenceRatio: Double
    let avgPainLast7: Double
    let painTrend: Double
    let completionRatio: Double
    let currentStreak: Int

    /// Returns nil below `minimumLogs` logs.
    init?(logs: [ExerciseLog], calendar: Calendar = .current, today: Date = Date()) {
        guard logs.count >= Self.minimumLogs else { return nil }

        let sorted = logs.sorted { $0.wrappedDate < $1.wrappedDate }
        guard let firstDate = sorted.first?.wrappedDate else { return nil }
        daysSinceStart = max(1, calendar.dateComponents([.day], from: firstDate, to: today).day ?? 1)

        let streak = StreakCalculator.summary(for: logs, calendar: calendar, today: today)
        adherenceRatio = streak.adherenceLast7Days
        currentStreak = streak.currentStreak

        let recent7 = sorted.suffix(7)
        avgPainLast7 = recent7.reduce(0.0) { $0 + Double($1.painLevel) } / Double(max(recent7.count, 1))

        let midpoint = sorted.count / 2
        let earlierHalf = sorted[0..<max(midpoint, 1)]
        let laterHalf = sorted[midpoint...]
        let earlierAvgPain = earlierHalf.reduce(0.0) { $0 + Double($1.painLevel) } / Double(max(earlierHalf.count, 1))
        let laterAvgPain = laterHalf.reduce(0.0) { $0 + Double($1.painLevel) } / Double(max(laterHalf.count, 1))
        painTrend = laterAvgPain - earlierAvgPain

        let completionRatios = sorted.map { log -> Double in
            guard let target = ExerciseLibrary.byID(log.wrappedExerciseID)?.targetReps, target > 0 else { return 1.0 }
            return Double(log.reps) / Double(target)
        }
        completionRatio = completionRatios.reduce(0, +) / Double(max(completionRatios.count, 1))
    }
}

struct RecoveryPrediction {
    /// Calibrated probability of not reaching a 30% pain reduction by week 6.
    let offTrackProbability: Double
    let needsAttention: Bool
    let rationale: String

    var onTrack: Bool { !needsAttention }
}

/// Runs RecoveryFeatures through RecoveryGoalClassifier.mlmodel and applies the
/// calibration and decision policy chosen in the simulation study
/// (MLTraining/README.md). The model was trained on simulated patients, so its
/// output is an estimate under the study's stated assumptions.
struct RecoveryPredictor {
    private let model: RecoveryGoalClassifier?

    init() {
        model = try? RecoveryGoalClassifier(configuration: MLModelConfiguration())
    }

    /// Returns nil when the app should stay quiet: fewer than 3 logs, or less
    /// than `RecoveryModelPolicy.warmUpDays` days of logging history.
    func predict(from logs: [ExerciseLog], calendar: Calendar = .current, today: Date = Date()) -> RecoveryPrediction? {
        guard let model,
              let features = RecoveryFeatures(logs: logs, calendar: calendar, today: today),
              features.daysSinceStart >= RecoveryModelPolicy.warmUpDays
        else { return nil }

        do {
            let input = RecoveryGoalClassifierInput(
                daysSinceStart: Int64(features.daysSinceStart),
                adherenceRatio: features.adherenceRatio,
                avgPainLast7: features.avgPainLast7,
                painTrend: features.painTrend,
                completionRatio: features.completionRatio,
                currentStreak: Int64(features.currentStreak)
            )
            let output = try model.prediction(input: input)
            guard let raw = output.offTrackProbability[1] else { return nil }
            let probability = RecoveryModelPolicy.calibrated(raw)
            let needsAttention = probability >= RecoveryModelPolicy.decisionThreshold

            return RecoveryPrediction(
                offTrackProbability: probability,
                needsAttention: needsAttention,
                rationale: rationale(needsAttention: needsAttention, features: features)
            )
        } catch {
            return nil
        }
    }

    private func rationale(needsAttention: Bool, features: RecoveryFeatures) -> String {
        let loggingConsistently = features.adherenceRatio >= 0.5
        if !needsAttention {
            // The estimate leans on the pain trend, so a patient can be on pace
            // with patchy logging: only credit adherence when it's there.
            switch (features.painTrend < 0, loggingConsistently) {
            case (true, true):
                return "Pain is trending down and adherence is strong."
            case (true, false):
                return "Pain is trending down — logging more often will make this estimate more reliable."
            case (false, true):
                return "You're logging sessions consistently."
            case (false, false):
                return "On pace so far — logging more often will make this estimate more reliable."
            }
        }
        if !loggingConsistently {
            return "Adherence has dropped — try logging sessions more consistently."
        }
        if features.painTrend > 0 {
            return "Pain levels are trending up — consider discussing with your therapist."
        }
        // Flagged despite steady sessions: in the study these were mostly
        // patients whose pain was easing too slowly, which calls for a plan
        // review rather than more effort.
        return "Pain isn't easing as quickly as expected — worth reviewing your plan with your therapist."
    }
}
