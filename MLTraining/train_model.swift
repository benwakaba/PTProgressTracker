#!/usr/bin/swift
// Trains RecoveryGoalClassifier.mlmodel from synthetic_recovery_data.csv using CreateML.
// Run on macOS with Xcode installed:
//   swift train_model.swift
// Output is written to ../PTProgressTracker/ML/RecoveryGoalClassifier.mlmodel

import Foundation
import CreateML

let scriptDir = URL(fileURLWithPath: CommandLine.arguments.first ?? ".").deletingLastPathComponent()
let csvURL = scriptDir.appendingPathComponent("synthetic_recovery_data.csv")
let outputURL = scriptDir
    .deletingLastPathComponent()
    .appendingPathComponent("PTProgressTracker/ML/RecoveryGoalClassifier.mlmodel")

print("Loading training data from \(csvURL.path)")
let data = try MLDataTable(contentsOf: csvURL)

let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 42)

print("Training MLBoostedTreeClassifier on \(trainingData.rows.count) rows...")
let classifier = try MLBoostedTreeClassifier(
    trainingData: trainingData,
    targetColumn: "onTrack"
)

let trainingAccuracy = (1.0 - classifier.trainingMetrics.classificationError) * 100
let validationAccuracy = (1.0 - classifier.validationMetrics.classificationError) * 100
print(String(format: "Training accuracy: %.1f%%", trainingAccuracy))
print(String(format: "Validation accuracy: %.1f%%", validationAccuracy))

let evaluation = classifier.evaluation(on: testingData)
let testAccuracy = (1.0 - evaluation.classificationError) * 100
print(String(format: "Held-out test accuracy: %.1f%%", testAccuracy))

let metadata = MLModelMetadata(
    author: "PT Progress Tracker",
    shortDescription: "Predicts whether a patient is on track to meet their PT recovery goal based on logged adherence, pain trend, completion ratio, and streak. Trained on synthetic data.",
    version: "1.0"
)

try FileManager.default.createDirectory(
    at: outputURL.deletingLastPathComponent(),
    withIntermediateDirectories: true
)
try classifier.write(to: outputURL, metadata: metadata)
print("Wrote model to \(outputURL.path)")
