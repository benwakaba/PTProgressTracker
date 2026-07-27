import SwiftUI

struct RootView: View {
    var body: some View {
        TabView {
            HomeView()
                .tabItem { Label("Home", systemImage: "house.fill") }

            ExerciseLibraryView()
                .tabItem { Label("Library", systemImage: "figure.strengthtraining.functional") }

            ProgressChartsView()
                .tabItem { Label("Progress", systemImage: "chart.line.uptrend.xyaxis") }

            NavigationStack {
                SessionHistoryView()
            }
            .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
        }
    }
}

#Preview {
    RootView()
        .environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
}
