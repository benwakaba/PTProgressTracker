import SwiftUI
import CoreData

struct HomeView: View {
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(key: "date", ascending: false)]
    )
    private var logs: FetchedResults<ExerciseLog>

    @State private var predictor = RecoveryPredictor()
    @State private var showExerciseCategoryPicker = false

    private var streak: StreakSummary {
        StreakCalculator.summary(for: Array(logs))
    }

    private var prediction: RecoveryPrediction? {
        predictor.predict(from: Array(logs))
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    streakCard
                    if let prediction {
                        PredictionCard(prediction: prediction)
                    }
                    quickLogSection
                    recentActivitySection
                }
                .padding()
            }
            .navigationTitle("PT Progress")
        }
    }

    private var streakCard: some View {
        HStack(spacing: 20) {
            VStack {
                Text("\(streak.currentStreak)")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                Text("day streak")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Divider().frame(height: 40)
            VStack {
                Text("\(streak.longestStreak)")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                Text("longest")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Divider().frame(height: 40)
            VStack {
                Text("\(Int(streak.adherenceLast7Days * 100))%")
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                Text("7-day adherence")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 20)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
    }

    private var quickLogSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Quick Log")
                .font(.headline)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 12) {
                    ForEach(ExerciseLibrary.all) { exercise in
                        NavigationLink {
                            ExerciseDetailView(exercise: exercise)
                        } label: {
                            VStack(spacing: 8) {
                                ExercisePhoto(exercise: exercise)
                                    .frame(width: 70, height: 70)
                                Text(exercise.name)
                                    .font(.caption.weight(.medium))
                                    .lineLimit(2)
                                    .multilineTextAlignment(.center)
                                    .frame(width: 80)
                                    .foregroundStyle(.primary)
                            }
                        }
                    }
                }
            }
        }
    }

    private var recentActivitySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Recent Activity")
                    .font(.headline)
                Spacer()
                NavigationLink("See all") {
                    SessionHistoryView()
                }
                .font(.subheadline)
            }
            if logs.isEmpty {
                Text("No sessions logged yet. Log your first exercise to start tracking progress.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14))
            } else {
                VStack(spacing: 8) {
                    ForEach(logs.prefix(5)) { log in
                        SessionRow(log: log)
                    }
                }
            }
        }
    }
}

private struct PredictionCard: View {
    let prediction: RecoveryPrediction

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: prediction.onTrack ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                .font(.title2)
                .foregroundStyle(prediction.onTrack ? .green : .orange)
            VStack(alignment: .leading, spacing: 4) {
                Text(prediction.onTrack ? "On track for your recovery goal" : "Recovery pace needs attention")
                    .font(.subheadline.weight(.semibold))
                Text(prediction.rationale)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Estimate from a model trained on simulated patients, not medical advice.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
        }
        .padding(14)
        .background((prediction.onTrack ? Color.green : Color.orange).opacity(0.12), in: RoundedRectangle(cornerRadius: 14))
    }
}

struct SessionRow: View {
    @ObservedObject var log: ExerciseLog

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(log.wrappedExerciseName)
                    .font(.subheadline.weight(.medium))
                Text(log.wrappedDate.formatted(date: .abbreviated, time: .shortened))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(log.sets)×\(log.reps)")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
            PainBadge(level: Int(log.painLevel))
        }
        .padding(10)
        .background(Color(.systemBackground), in: RoundedRectangle(cornerRadius: 12))
        .shadow(color: .black.opacity(0.04), radius: 3, y: 1)
    }
}

struct PainBadge: View {
    let level: Int

    private var color: Color {
        switch level {
        case 0...2: return .green
        case 3...5: return .yellow
        case 6...7: return .orange
        default: return .red
        }
    }

    var body: some View {
        Text("Pain \(level)")
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.2), in: Capsule())
            .foregroundStyle(color)
    }
}

#Preview {
    HomeView()
        .environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
}
