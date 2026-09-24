import XCTest
import CoreData
@testable import PTProgressTracker

final class PTProgressTrackerTests: XCTestCase {
    private var context: NSManagedObjectContext!

    override func setUp() {
        super.setUp()
        context = PersistenceController(inMemory: true).container.viewContext
    }

    private func makeLog(daysAgo: Int, pain: Int32, reps: Int32 = 10, sets: Int32 = 3) -> ExerciseLog {
        let log = ExerciseLog(context: context)
        log.id = UUID()
        log.exerciseID = "knee-extension"
        log.exerciseName = "Seated Knee Extension"
        log.date = Calendar.current.date(byAdding: .day, value: -daysAgo, to: Date())!
        log.reps = reps
        log.sets = sets
        log.painLevel = pain
        log.notes = ""
        log.createdAt = Date()
        return log
    }

    func testStreakCountsConsecutiveDays() {
        let logs = [makeLog(daysAgo: 0, pain: 3), makeLog(daysAgo: 1, pain: 4), makeLog(daysAgo: 2, pain: 5)]
        let summary = StreakCalculator.summary(for: logs)
        XCTAssertEqual(summary.currentStreak, 3)
    }

    func testStreakBreaksOnGap() {
        let logs = [makeLog(daysAgo: 0, pain: 3), makeLog(daysAgo: 2, pain: 4)]
        let summary = StreakCalculator.summary(for: logs)
        XCTAssertEqual(summary.currentStreak, 1)
    }

    func testExerciseLibraryLookup() {
        XCTAssertEqual(ExerciseLibrary.byID("knee-extension")?.name, "Seated Knee Extension")
        XCTAssertNil(ExerciseLibrary.byID("nonexistent"))
    }

    func testRecoveryPredictorReturnsNilWithoutEnoughLogs() {
        let predictor = RecoveryPredictor()
        let logs = [makeLog(daysAgo: 0, pain: 3)]
        XCTAssertNil(predictor.predict(from: logs))
    }

    func testRecoveryPredictorProducesPredictionWithSufficientLogs() {
        let predictor = RecoveryPredictor()
        let logs = (0..<10).map { makeLog(daysAgo: $0, pain: Int32(max(1, 6 - $0 / 2))) }
        let prediction = predictor.predict(from: logs)
        XCTAssertNotNil(prediction)
    }

    func testRecoveryPredictorWaitsForAWeekOfHistory() {
        let predictor = RecoveryPredictor()
        let logs = (0..<5).map { makeLog(daysAgo: $0, pain: 4) }
        XCTAssertNil(predictor.predict(from: logs))
    }

    func testRecoveryPredictorFlagsDecliningPatientButNotImprovingOne() throws {
        let predictor = RecoveryPredictor()

        // Two weeks of daily sessions, full reps, pain falling from 7 to 1.
        let improving = (0..<14).map { makeLog(daysAgo: $0, pain: Int32(1 + $0 / 2), reps: 12) }
        let good = try XCTUnwrap(predictor.predict(from: improving))
        XCTAssertFalse(good.needsAttention)
        XCTAssertLessThan(good.offTrackProbability, RecoveryModelPolicy.decisionThreshold)

        // Four sessions a fortnight ago, half the reps, pain rising, nothing since.
        let declining = [(13, 4), (12, 5), (11, 6), (10, 7)].map { makeLog(daysAgo: $0.0, pain: Int32($0.1), reps: 6) }
        let bad = try XCTUnwrap(predictor.predict(from: declining))
        XCTAssertTrue(bad.needsAttention)
        XCTAssertEqual(bad.rationale, "Adherence has dropped — try logging sessions more consistently.")
    }

    func testOnTrackRationaleOnlyCreditsAdherenceWhenItIsThere() throws {
        let predictor = RecoveryPredictor()
        // Pain falling from 5 to 1, but nothing logged in the past week.
        let logs = [(12, 5), (11, 4), (10, 3), (9, 2), (8, 1)].map { makeLog(daysAgo: $0.0, pain: Int32($0.1)) }
        let prediction = try XCTUnwrap(predictor.predict(from: logs))
        XCTAssertTrue(prediction.onTrack)
        XCTAssertEqual(prediction.rationale, "Pain is trending down — logging more often will make this estimate more reliable.")
    }
}
