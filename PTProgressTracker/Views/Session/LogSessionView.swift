import SwiftUI

struct LogSessionView: View {
    let exercise: Exercise

    @Environment(\.managedObjectContext) private var context
    @Environment(\.dismiss) private var dismiss

    @State private var reps: Int
    @State private var sets: Int
    @State private var painLevel: Double = 3
    @State private var notes: String = ""
    @State private var date: Date = Date()

    init(exercise: Exercise) {
        self.exercise = exercise
        _reps = State(initialValue: exercise.targetReps)
        _sets = State(initialValue: exercise.targetSets)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Session") {
                    DatePicker("Date", selection: $date, displayedComponents: [.date, .hourAndMinute])
                    Stepper("Sets: \(sets)", value: $sets, in: 1...10)
                    Stepper("Reps: \(reps)", value: $reps, in: 1...50)
                }

                Section("Pain level (0 = none, 10 = severe)") {
                    HStack {
                        Text("\(Int(painLevel))")
                            .font(.title3.weight(.semibold))
                            .frame(width: 32)
                            .foregroundStyle(painColor)
                        Slider(value: $painLevel, in: 0...10, step: 1)
                    }
                }

                Section("Notes") {
                    TextField("Optional notes about this session", text: $notes, axis: .vertical)
                        .lineLimit(3...6)
                }
            }
            .navigationTitle("Log Session")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                        .fontWeight(.semibold)
                }
            }
        }
    }

    private var painColor: Color {
        switch Int(painLevel) {
        case 0...2: return .green
        case 3...5: return .yellow
        case 6...7: return .orange
        default: return .red
        }
    }

    private func save() {
        let log = ExerciseLog(context: context)
        log.id = UUID()
        log.exerciseID = exercise.id
        log.exerciseName = exercise.name
        log.date = date
        log.reps = Int32(reps)
        log.sets = Int32(sets)
        log.painLevel = Int32(painLevel)
        log.notes = notes
        log.createdAt = Date()

        do {
            try context.save()
            dismiss()
        } catch {
            assertionFailure("Failed to save log: \(error.localizedDescription)")
        }
    }
}

#Preview {
    LogSessionView(exercise: ExerciseLibrary.all[0])
        .environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
}
