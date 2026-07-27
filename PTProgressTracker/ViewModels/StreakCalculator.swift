import Foundation

struct StreakSummary {
    let currentStreak: Int
    let longestStreak: Int
    let last7DaysLogged: Set<Date>
    let adherenceLast7Days: Double // 0...1
}

enum StreakCalculator {
    static func summary(for logs: [ExerciseLog], calendar: Calendar = .current, today: Date = Date()) -> StreakSummary {
        let loggedDays = Set(logs.map { calendar.startOfDay(for: $0.wrappedDate) })
        let todayStart = calendar.startOfDay(for: today)

        var currentStreak = 0
        var cursor = todayStart
        if !loggedDays.contains(cursor) {
            cursor = calendar.date(byAdding: .day, value: -1, to: cursor) ?? cursor
        }
        while loggedDays.contains(cursor) {
            currentStreak += 1
            guard let previous = calendar.date(byAdding: .day, value: -1, to: cursor) else { break }
            cursor = previous
        }

        var longestStreak = 0
        var runningStreak = 0
        let sortedDays = loggedDays.sorted()
        var previousDay: Date?
        for day in sortedDays {
            if let previousDay, calendar.date(byAdding: .day, value: 1, to: previousDay) == day {
                runningStreak += 1
            } else {
                runningStreak = 1
            }
            longestStreak = max(longestStreak, runningStreak)
            previousDay = day
        }

        var last7: Set<Date> = []
        for offset in 0..<7 {
            if let day = calendar.date(byAdding: .day, value: -offset, to: todayStart), loggedDays.contains(day) {
                last7.insert(day)
            }
        }
        let adherence = Double(last7.count) / 7.0

        return StreakSummary(
            currentStreak: currentStreak,
            longestStreak: longestStreak,
            last7DaysLogged: last7,
            adherenceLast7Days: adherence
        )
    }
}
