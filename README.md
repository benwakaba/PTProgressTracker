# PT Progress Tracker

A patient-facing iOS app for following prescribed physical therapy plans, logging
session data, and visualizing recovery over time.

## Stack

- **SwiftUI** for all UI (iOS 17+)
- **Core Data** for local persistence of exercise logs (`PTProgressTracker.xcdatamodeld`)
- **Swift Charts** for pain-trend, daily-volume, and exercise-breakdown visualizations
- **CoreML** for an on-device "recovery on track?" prediction, trained via **CreateML**
  on synthetic data

## Project layout

```
PTProgressTracker/
  App/                   App entry point
  CoreData/              PersistenceController
  Models/                Exercise library (static data) + Core Data model extensions
  ML/                    RecoveryPredictor.swift + trained RecoveryGoalClassifier.mlmodel
  Views/
    Shared/    RootView (TabView), HomeView, AnimatedExerciseDiagram
    Library/   Exercise browsing + detail
    Session/   Log-a-session form, session history
    Progress/  Swift Charts dashboards
PTProgressTracker.xcdatamodeld/   Core Data model
PTProgressTrackerTests/           Unit tests (streak logic, ML wrapper)
MLTraining/
  generate_synthetic_data.py      Synthetic patient logging data generator
  train_model.swift               CreateML training script -> RecoveryGoalClassifier.mlmodel
project.yml                       XcodeGen spec (this repo has no committed .xcodeproj)
```

## Opening the project

The `.xcodeproj` is generated, not committed. Regenerate it with
[XcodeGen](https://github.com/yonaskolb/XcodeGen) any time you add/remove files:

```bash
brew install xcodegen   # one-time
cd PTProgressTracker
xcodegen generate
open PTProgressTracker.xcodeproj
```

## Retraining the CoreML model

`PTProgressTracker/ML/RecoveryGoalClassifier.mlmodel` is already trained and checked in
(~77% held-out accuracy on synthetic data). To regenerate it:

```bash
cd MLTraining
python3 generate_synthetic_data.py     # writes synthetic_recovery_data.csv
swift train_model.swift                # trains + writes ../PTProgressTracker/ML/RecoveryGoalClassifier.mlmodel
```

`RecoveryPredictor.swift` computes the model's six input features (adherence ratio,
pain trend, completion ratio, streak, days since start) from a patient's logged
`ExerciseLog` history and surfaces an on-track / needs-attention card on the Home tab.

## Design notes / deliberate simplifications

- **Animated exercise diagrams** use a small dependency-free `Canvas`-based
  stick-figure animator (`AnimatedExerciseDiagram.swift`) instead of Lottie, so the
  build has no external SPM dependency to fetch. Swapping in real Lottie JSON assets
  later only requires changing the body of that one view — call sites depend only on
  `DiagramKind`.
- **CloudKit sync** was left out (it was marked optional in the brief) since it needs
  an paid Apple Developer account + container to actually test; `PersistenceController`
  is structured so swapping `NSPersistentContainer` for `NSPersistentCloudKitContainer`
  is a small, isolated change if you want to add it later.

## Verification performed

- `xcodebuild ... build` — clean build succeeds for the iOS Simulator SDK (Debug config).
- Installed and launched the built `.app` on a booted iPhone 15 simulator via
  `xcrun simctl install/launch` and confirmed the Home dashboard renders correctly
  (streak card, quick-log animated diagrams, empty-state copy) via screenshot.
- `PTProgressTrackerTests` covers streak-calculation edge cases and the
  `RecoveryPredictor` feature pipeline (nil below 3 logs, non-nil above).

Note: this machine's Xcode/CoreSimulator device-discovery got into a stale state
mid-session (`iOS 26.5 is not installed` / CoreSimulator version-mismatch errors from
`xcodebuild -destination`), which blocked running `xcodebuild test` and the Simulator
MCP tool for the remainder of the session — `xcrun simctl` still worked directly. If
you hit the same thing in Xcode, `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
followed by opening Xcode.app once (to let it finish first-launch component setup)
is the usual fix.
