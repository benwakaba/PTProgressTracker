import SwiftUI
import Charts
import CoreData

private struct DailyAggregate: Identifiable {
    let day: Date
    let totalReps: Int
    let avgPain: Double
    var id: Date { day }
}

struct ProgressChartsView: View {
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(key: "date", ascending: true)]
    )
    private var logs: FetchedResults<ExerciseLog>

    @State private var rangeDays: Int = 30

    private var calendar: Calendar { .current }

    private var filteredLogs: [ExerciseLog] {
        guard let cutoff = calendar.date(byAdding: .day, value: -rangeDays, to: Date()) else {
            return Array(logs)
        }
        return logs.filter { $0.wrappedDate >= cutoff }
    }

    private var dailyAggregates: [DailyAggregate] {
        let grouped = Dictionary(grouping: filteredLogs) { calendar.startOfDay(for: $0.wrappedDate) }
        return grouped.map { day, entries in
            let totalReps = entries.reduce(0) { $0 + Int($1.reps) * Int($1.sets) }
            let avgPain = entries.reduce(0.0) { $0 + Double($1.painLevel) } / Double(entries.count)
            return DailyAggregate(day: day, totalReps: totalReps, avgPain: avgPain)
        }.sorted { $0.day < $1.day }
    }

    private var exerciseFrequency: [(name: String, count: Int)] {
        let grouped = Dictionary(grouping: filteredLogs) { $0.wrappedExerciseName }
        return grouped.map { ($0.key, $0.value.count) }.sorted { $0.1 > $1.1 }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    Picker("Range", selection: $rangeDays) {
                        Text("7d").tag(7)
                        Text("30d").tag(30)
                        Text("90d").tag(90)
                    }
                    .pickerStyle(.segmented)

                    if filteredLogs.isEmpty {
                        ContentUnavailableView(
                            "No Data Yet",
                            systemImage: "chart.line.uptrend.xyaxis",
                            description: Text("Log a few sessions to see your progress charts.")
                        )
                        .frame(height: 300)
                    } else {
                        painTrendChart
                        volumeChart
                        exerciseBreakdownChart
                    }
                }
                .padding()
            }
            .navigationTitle("Progress")
        }
    }

    private var painTrendChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Pain Level Trend")
                .font(.headline)
            Text("Lower is better — average reported pain per day")
                .font(.caption)
                .foregroundStyle(.secondary)
            Chart(dailyAggregates) { point in
                LineMark(x: .value("Day", point.day, unit: .day), y: .value("Pain", point.avgPain))
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(.orange)
                PointMark(x: .value("Day", point.day, unit: .day), y: .value("Pain", point.avgPain))
                    .foregroundStyle(.orange)
            }
            .chartYScale(domain: 0...10)
            .frame(height: 180)
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
    }

    private var volumeChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Daily Volume")
                .font(.headline)
            Text("Total reps completed (reps × sets) per day")
                .font(.caption)
                .foregroundStyle(.secondary)
            Chart(dailyAggregates) { point in
                BarMark(x: .value("Day", point.day, unit: .day), y: .value("Reps", point.totalReps))
                    .foregroundStyle(.blue.gradient)
            }
            .frame(height: 180)
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
    }

    private var exerciseBreakdownChart: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Exercise Breakdown")
                .font(.headline)
            Text("Sessions logged by exercise")
                .font(.caption)
                .foregroundStyle(.secondary)
            Chart(exerciseFrequency, id: \.name) { item in
                BarMark(x: .value("Sessions", item.count), y: .value("Exercise", item.name))
                    .foregroundStyle(.teal.gradient)
            }
            .frame(height: CGFloat(exerciseFrequency.count) * 36 + 20)
        }
        .padding()
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 16))
    }
}

#Preview {
    ProgressChartsView()
        .environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
}
