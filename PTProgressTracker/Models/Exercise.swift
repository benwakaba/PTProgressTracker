import SwiftUI

enum ExerciseCategory: String, CaseIterable, Identifiable {
    case knee = "Knee"
    case shoulder = "Shoulder"
    case back = "Back"
    case ankle = "Ankle"
    case hip = "Hip"

    var id: String { rawValue }

    var tint: Color {
        switch self {
        case .knee: return .blue
        case .shoulder: return .orange
        case .back: return .purple
        case .ankle: return .teal
        case .hip: return .pink
        }
    }
}

enum DiagramKind {
    case legExtension
    case shoulderRaise
    case pelvicTilt
    case ankleCircle
    case bridge
    case wallSlide
    case heelSlide
    case clamshell
}

struct Exercise: Identifiable, Hashable {
    let id: String
    let name: String
    let category: ExerciseCategory
    let instructions: [String]
    let targetReps: Int
    let targetSets: Int
    let diagram: DiagramKind
    /// SF Symbol used as the exercise's photo-style thumbnail/hero image.
    let photoSymbol: String

    static func == (lhs: Exercise, rhs: Exercise) -> Bool { lhs.id == rhs.id }
    func hash(into hasher: inout Hasher) { hasher.combine(id) }
}

enum ExerciseLibrary {
    static let all: [Exercise] = [
        Exercise(
            id: "knee-extension",
            name: "Seated Knee Extension",
            category: .knee,
            instructions: [
                "Sit tall in a chair with feet flat on the floor.",
                "Slowly straighten one knee until the leg is level.",
                "Hold for 3 seconds, then lower with control.",
                "Keep your back straight throughout the movement."
            ],
            targetReps: 12,
            targetSets: 3,
            diagram: .legExtension,
            photoSymbol: "figure.strengthtraining.traditional"
        ),
        Exercise(
            id: "shoulder-flexion",
            name: "Shoulder Flexion Raise",
            category: .shoulder,
            instructions: [
                "Stand with arms relaxed at your sides.",
                "Raise your arm forward and up toward the ceiling.",
                "Go only as high as comfortable, avoid shrugging.",
                "Lower slowly back to the start."
            ],
            targetReps: 10,
            targetSets: 3,
            diagram: .shoulderRaise,
            photoSymbol: "figure.arms.open"
        ),
        Exercise(
            id: "pelvic-tilt",
            name: "Pelvic Tilt",
            category: .back,
            instructions: [
                "Lie on your back with knees bent, feet flat.",
                "Flatten your lower back into the floor by tightening your abdominals.",
                "Hold for 5 seconds, then release.",
                "Breathe normally throughout."
            ],
            targetReps: 10,
            targetSets: 2,
            diagram: .pelvicTilt,
            photoSymbol: "figure.core.training"
        ),
        Exercise(
            id: "ankle-circles",
            name: "Ankle Circles",
            category: .ankle,
            instructions: [
                "Sit with your leg extended, heel resting on the floor.",
                "Rotate your ankle in a slow, full circle.",
                "Complete circles clockwise, then counter-clockwise.",
                "Keep the motion smooth and pain-free."
            ],
            targetReps: 15,
            targetSets: 2,
            diagram: .ankleCircle,
            photoSymbol: "figure.flexibility"
        ),
        Exercise(
            id: "glute-bridge",
            name: "Glute Bridge",
            category: .hip,
            instructions: [
                "Lie on your back, knees bent, feet hip-width apart.",
                "Squeeze your glutes and lift your hips toward the ceiling.",
                "Hold briefly at the top, keeping hips level.",
                "Lower slowly back down."
            ],
            targetReps: 12,
            targetSets: 3,
            diagram: .bridge,
            photoSymbol: "figure.cooldown"
        ),
        Exercise(
            id: "wall-slide",
            name: "Wall Slide",
            category: .shoulder,
            instructions: [
                "Stand with your back against a wall, arms in a goalpost shape.",
                "Slide your arms up the wall as far as comfortable.",
                "Keep your wrists and elbows touching the wall.",
                "Slide back down with control."
            ],
            targetReps: 10,
            targetSets: 3,
            diagram: .wallSlide,
            photoSymbol: "figure.stand"
        ),
        Exercise(
            id: "heel-slide",
            name: "Heel Slide",
            category: .knee,
            instructions: [
                "Lie on your back with legs extended.",
                "Slide one heel toward your glutes, bending the knee.",
                "Slide back out to a straight leg.",
                "Keep the motion slow and controlled."
            ],
            targetReps: 12,
            targetSets: 3,
            diagram: .heelSlide,
            photoSymbol: "figure.walk.motion"
        ),
        Exercise(
            id: "clamshell",
            name: "Clamshell",
            category: .hip,
            instructions: [
                "Lie on your side with knees bent, feet together.",
                "Keeping feet touching, lift your top knee like a clamshell opening.",
                "Hold briefly, then lower with control.",
                "Keep your hips stacked throughout."
            ],
            targetReps: 15,
            targetSets: 3,
            diagram: .clamshell,
            photoSymbol: "figure.pilates"
        )
    ]

    static func byID(_ id: String) -> Exercise? {
        all.first { $0.id == id }
    }
}
