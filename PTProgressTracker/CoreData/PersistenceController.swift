import CoreData

struct PersistenceController {
    static let shared = PersistenceController()

    static var preview: PersistenceController = {
        let controller = PersistenceController(inMemory: true)
        controller.seedPreviewData()
        return controller
    }()

    let container: NSPersistentContainer

    init(inMemory: Bool = false) {
        container = NSPersistentContainer(name: "PTProgressTracker")
        if inMemory {
            container.persistentStoreDescriptions.first?.url = URL(fileURLWithPath: "/dev/null")
        }
        container.loadPersistentStores { _, error in
            if let error {
                fatalError("Unresolved Core Data error: \(error.localizedDescription)")
            }
        }
        container.viewContext.automaticallyMergesChangesFromParent = true
    }

    func save() {
        let context = container.viewContext
        guard context.hasChanges else { return }
        do {
            try context.save()
        } catch {
            assertionFailure("Failed to save context: \(error.localizedDescription)")
        }
    }

    private func seedPreviewData() {
        let context = container.viewContext
        let calendar = Calendar.current
        for dayOffset in 0..<14 {
            guard let date = calendar.date(byAdding: .day, value: -dayOffset, to: Date()) else { continue }
            let log = ExerciseLog(context: context)
            log.id = UUID()
            log.exerciseID = ExerciseLibrary.all[dayOffset % ExerciseLibrary.all.count].id
            log.exerciseName = ExerciseLibrary.all[dayOffset % ExerciseLibrary.all.count].name
            log.date = date
            log.reps = Int32(10 + dayOffset % 5)
            log.sets = 3
            log.painLevel = Int32(max(1, 6 - dayOffset / 3))
            log.notes = ""
            log.createdAt = date
        }
        try? context.save()
    }
}
