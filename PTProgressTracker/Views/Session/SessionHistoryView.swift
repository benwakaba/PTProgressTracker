import SwiftUI
import CoreData

struct SessionHistoryView: View {
    @Environment(\.managedObjectContext) private var context
    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(key: "date", ascending: false)]
    )
    private var logs: FetchedResults<ExerciseLog>

    var body: some View {
        List {
            ForEach(logs) { log in
                SessionRow(log: log)
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
            }
            .onDelete(perform: delete)
        }
        .listStyle(.plain)
        .navigationTitle("Session History")
        .overlay {
            if logs.isEmpty {
                ContentUnavailableView(
                    "No Sessions Yet",
                    systemImage: "list.clipboard",
                    description: Text("Logged sessions will appear here.")
                )
            }
        }
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            context.delete(logs[index])
        }
        try? context.save()
    }
}

#Preview {
    NavigationStack {
        SessionHistoryView()
            .environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
    }
}
