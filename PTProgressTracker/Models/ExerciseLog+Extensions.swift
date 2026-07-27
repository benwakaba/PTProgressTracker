import Foundation

/// Core Data's default codegen makes even non-optional model attributes of
/// reference type (String/Date/UUID) produce Optional Swift properties.
/// These wrappers give call sites non-optional access without scattering
/// `?? ...` fallbacks throughout the views and view models.
extension ExerciseLog {
    var wrappedDate: Date { date ?? createdAt ?? Date.distantPast }
    var wrappedExerciseName: String { exerciseName ?? "Exercise" }
    var wrappedExerciseID: String { exerciseID ?? "" }
    var wrappedNotes: String { notes ?? "" }
}
