import XCTest
import CoreData
@testable import PTProgressTracker

/// Holds RecoveryFeatures to the fixture MLTraining/features.py wrote. If this
/// fails, the app is feeding the model inputs computed differently from the
/// ones it was trained on. Regenerate with MLTraining/make_parity_fixture.py
/// only after making the same change on both sides.
final class FeatureParityTests: XCTestCase {
    private struct Fixture: Decodable {
        let cases: [Case]
    }

    private struct Case: Decodable {
        let name: String
        let today: String
        let logs: [Entry]
        let expected: Expected?
    }

    private struct Entry: Decodable {
        let timestamp: String
        let exerciseID: String
        let reps: Int32
        let painLevel: Int32
    }

    private struct Expected: Decodable {
        let daysSinceStart: Int
        let adherenceRatio: Double
        let avgPainLast7: Double
        let painTrend: Double
        let completionRatio: Double
        let currentStreak: Int
    }

    func testFeaturesMatchPythonImplementation() throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "feature_parity_cases", withExtension: "json"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        XCTAssertGreaterThanOrEqual(fixture.cases.count, 20)

        var utc = Calendar(identifier: .gregorian)
        utc.timeZone = try XCTUnwrap(TimeZone(identifier: "UTC"))
        let iso = ISO8601DateFormatter()
        let context = PersistenceController(inMemory: true).container.viewContext

        for testCase in fixture.cases {
            let logs = try testCase.logs.map { entry -> ExerciseLog in
                let log = ExerciseLog(context: context)
                log.id = UUID()
                log.exerciseID = entry.exerciseID
                log.exerciseName = entry.exerciseID
                log.date = try XCTUnwrap(iso.date(from: entry.timestamp))
                log.reps = entry.reps
                log.sets = 3
                log.painLevel = entry.painLevel
                return log
            }
            let today = try XCTUnwrap(iso.date(from: testCase.today))
            let features = RecoveryFeatures(logs: logs, calendar: utc, today: today)

            guard let expected = testCase.expected else {
                XCTAssertNil(features, testCase.name)
                continue
            }
            let actual = try XCTUnwrap(features, testCase.name)
            XCTAssertEqual(actual.daysSinceStart, expected.daysSinceStart, testCase.name)
            XCTAssertEqual(actual.currentStreak, expected.currentStreak, testCase.name)
            XCTAssertEqual(actual.adherenceRatio, expected.adherenceRatio, accuracy: 1e-9, testCase.name)
            XCTAssertEqual(actual.avgPainLast7, expected.avgPainLast7, accuracy: 1e-9, testCase.name)
            XCTAssertEqual(actual.painTrend, expected.painTrend, accuracy: 1e-9, testCase.name)
            XCTAssertEqual(actual.completionRatio, expected.completionRatio, accuracy: 1e-9, testCase.name)
        }
    }
}
