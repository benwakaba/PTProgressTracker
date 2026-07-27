import SwiftUI

struct ExerciseDetailView: View {
    let exercise: Exercise
    @State private var showLogSheet = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                ExercisePhoto(exercise: exercise, symbolScale: 0.34, aspectRatio: nil)
                    .frame(height: 180)
                    .frame(maxWidth: .infinity)

                VStack(alignment: .leading, spacing: 8) {
                    Text("Animated Form Guide")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.secondary)
                    AnimatedExerciseDiagram(kind: exercise.diagram, tint: exercise.category.tint)
                        .frame(height: 180)
                        .frame(maxWidth: .infinity)
                        .background(exercise.category.tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 20))
                }

                HStack {
                    Label(exercise.category.rawValue, systemImage: "figure.walk")
                        .font(.subheadline.weight(.medium))
                        .foregroundStyle(exercise.category.tint)
                    Spacer()
                    Text("Target: \(exercise.targetSets) sets × \(exercise.targetReps) reps")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 10) {
                    Text("How to perform")
                        .font(.headline)
                    ForEach(Array(exercise.instructions.enumerated()), id: \.offset) { index, step in
                        HStack(alignment: .top, spacing: 10) {
                            Text("\(index + 1)")
                                .font(.caption.weight(.bold))
                                .frame(width: 22, height: 22)
                                .background(exercise.category.tint, in: Circle())
                                .foregroundStyle(.white)
                            Text(step)
                                .font(.body)
                            Spacer()
                        }
                    }
                }

                Button {
                    showLogSheet = true
                } label: {
                    Label("Log This Session", systemImage: "plus.circle.fill")
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .padding()
                }
                .buttonStyle(.borderedProminent)
                .tint(exercise.category.tint)
            }
            .padding()
        }
        .navigationTitle(exercise.name)
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showLogSheet) {
            LogSessionView(exercise: exercise)
        }
    }
}

#Preview {
    NavigationStack {
        ExerciseDetailView(exercise: ExerciseLibrary.all[0])
    }
}
