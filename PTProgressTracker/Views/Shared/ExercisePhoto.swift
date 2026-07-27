import SwiftUI

/// A photo-style thumbnail/hero image for an exercise, built from an SF
/// Symbol on a tinted gradient card — matches the exercise's category color
/// and avoids bundling external/copyrighted photography.
struct ExercisePhoto: View {
    let exercise: Exercise
    var symbolScale: CGFloat = 0.46
    /// Pass nil to let the parent control sizing via `.frame` instead of a fixed square.
    var aspectRatio: CGFloat? = 1

    var body: some View {
        GeometryReader { geo in
            ZStack {
                LinearGradient(
                    colors: [exercise.category.tint, exercise.category.tint.opacity(0.6)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                Image(systemName: exercise.photoSymbol)
                    .font(.system(size: min(geo.size.width, geo.size.height) * symbolScale, weight: .medium))
                    .foregroundStyle(.white)
                    .shadow(color: .black.opacity(0.15), radius: 3, y: 2)
            }
        }
        .modifier(OptionalAspectRatio(ratio: aspectRatio))
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .accessibilityLabel(Text("Photo of \(exercise.name)"))
    }
}

private struct OptionalAspectRatio: ViewModifier {
    let ratio: CGFloat?
    func body(content: Content) -> some View {
        if let ratio {
            content.aspectRatio(ratio, contentMode: .fit)
        } else {
            content
        }
    }
}

#Preview {
    ExercisePhoto(exercise: ExerciseLibrary.all[0])
        .frame(width: 120, height: 120)
        .padding()
}
