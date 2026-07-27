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
}
