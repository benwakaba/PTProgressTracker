import SwiftUI

/// Lightweight, dependency-free animated stick-figure diagrams that loop to
/// illustrate exercise form. Swap the body of this view for a Lottie
/// `LottieView(animation: .named(...))` (via the lottie-ios SPM package) if
/// richer illustrator-authored animations are added later — the call sites
/// in ExerciseDetailView / ExerciseRow only depend on `DiagramKind`.
struct AnimatedExerciseDiagram: View {
    let kind: DiagramKind
    let tint: Color

    @State private var phase: Bool = false

    var body: some View {
        Canvas { context, size in
            draw(kind: kind, phase: phase, size: size, tint: tint, context: &context)
        }
        .onAppear { phase = false }
        .task {
            withAnimation(.easeInOut(duration: 1.1).repeatForever(autoreverses: true)) {
                phase = true
            }
        }
        .accessibilityLabel(Text("Animated demonstration of \(kind.description)"))
    }

    private func draw(kind: DiagramKind, phase: Bool, size: CGSize, tint: Color, context: inout GraphicsContext) {
        let w = size.width, h = size.height
        let stroke = StrokeStyle(lineWidth: max(3, w * 0.03), lineCap: .round, lineJoin: .round)
        let bodyColor = GraphicsContext.Shading.color(tint)
        let jointColor = GraphicsContext.Shading.color(tint.opacity(0.9))

        func line(_ from: CGPoint, _ to: CGPoint) {
            var path = Path()
            path.move(to: from)
            path.addLine(to: to)
            context.stroke(path, with: bodyColor, style: stroke)
        }
        func dot(_ p: CGPoint, r: CGFloat = 5) {
            context.fill(Path(ellipseIn: CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)), with: jointColor)
        }

        let cx = w * 0.5
        switch kind {
        case .legExtension:
            let hip = CGPoint(x: cx - w * 0.15, y: h * 0.35)
            let knee = CGPoint(x: cx + w * 0.05, y: h * 0.55)
            let footAngle: CGFloat = phase ? -0.05 : 0.55
            let foot = CGPoint(x: knee.x + cos(footAngle) * w * 0.28, y: knee.y + sin(footAngle) * h * 0.28)
            line(CGPoint(x: cx - w * 0.15, y: h * 0.1), hip)
            line(hip, knee); dot(hip); dot(knee)
            line(knee, foot); dot(foot)

        case .shoulderRaise:
            let shoulder = CGPoint(x: cx, y: h * 0.3)
            let armAngle: CGFloat = phase ? -1.4 : 1.4
            let hand = CGPoint(x: shoulder.x + cos(armAngle) * w * 0.35, y: shoulder.y + sin(armAngle) * h * 0.35)
            line(CGPoint(x: cx, y: h * 0.1), shoulder)
            line(shoulder, CGPoint(x: cx, y: h * 0.75))
            line(shoulder, hand); dot(shoulder); dot(hand)

        case .pelvicTilt:
            let tiltOffset: CGFloat = phase ? -h * 0.06 : 0
            let hip = CGPoint(x: cx, y: h * 0.55 + tiltOffset)
            line(CGPoint(x: cx - w * 0.25, y: h * 0.6), hip)
            line(hip, CGPoint(x: cx + w * 0.25, y: h * 0.6))
            dot(hip)

        case .ankleCircle:
            let center = CGPoint(x: cx, y: h * 0.55)
            let angle = phase ? CGFloat.pi * 1.8 : 0
            let tip = CGPoint(x: center.x + cos(angle) * w * 0.25, y: center.y + sin(angle) * h * 0.15)
            line(CGPoint(x: cx, y: h * 0.2), center)
            line(center, tip); dot(center); dot(tip)

        case .bridge:
            let liftOffset: CGFloat = phase ? -h * 0.15 : 0
            let hip = CGPoint(x: cx, y: h * 0.5 + liftOffset)
            line(CGPoint(x: cx - w * 0.3, y: h * 0.65), hip)
            line(hip, CGPoint(x: cx + w * 0.3, y: h * 0.65))
            dot(hip)

        case .wallSlide:
            let shoulder = CGPoint(x: cx, y: h * 0.35)
            let slideAngle: CGFloat = phase ? -1.55 : -0.7
            let hand = CGPoint(x: shoulder.x + cos(slideAngle) * w * 0.3, y: shoulder.y + sin(slideAngle) * h * 0.3)
            line(CGPoint(x: cx, y: h * 0.15), shoulder)
            line(shoulder, hand); dot(shoulder); dot(hand)

        case .heelSlide:
            let hip = CGPoint(x: cx - w * 0.2, y: h * 0.4)
            let kneeBend: CGFloat = phase ? 0.9 : 0.15
            let knee = CGPoint(x: hip.x + cos(kneeBend) * w * 0.3, y: hip.y + sin(kneeBend) * h * 0.3)
            line(CGPoint(x: cx - w * 0.2, y: h * 0.15), hip)
            line(hip, knee); dot(hip); dot(knee)

        case .clamshell:
            let hip = CGPoint(x: cx, y: h * 0.5)
            let kneeAngle: CGFloat = phase ? 0.9 : 0.3
            let knee = CGPoint(x: hip.x + cos(kneeAngle) * w * 0.3, y: hip.y - sin(kneeAngle) * h * 0.1)
            line(CGPoint(x: cx, y: h * 0.2), hip)
            line(hip, knee); dot(hip); dot(knee)
        }
    }
}

extension DiagramKind {
    var description: String {
        switch self {
        case .legExtension: return "seated knee extension"
        case .shoulderRaise: return "shoulder flexion raise"
        case .pelvicTilt: return "pelvic tilt"
        case .ankleCircle: return "ankle circles"
        case .bridge: return "glute bridge"
        case .wallSlide: return "wall slide"
        case .heelSlide: return "heel slide"
        case .clamshell: return "clamshell"
        }
    }
}
