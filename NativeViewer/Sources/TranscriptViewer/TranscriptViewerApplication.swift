import AppKit
import SwiftUI

@main
@MainActor
final class TranscriptViewerApplication: NSObject, NSApplicationDelegate {
    private static var retainedDelegate: TranscriptViewerApplication?

    private var window: NSWindow?
    private var model: LibraryViewModel?

    static func main() {
        let app = NSApplication.shared
        let delegate = TranscriptViewerApplication()
        retainedDelegate = delegate
        app.delegate = delegate
        app.setActivationPolicy(.regular)
        app.run()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        let initialPath = CommandLine.arguments.dropFirst().first
        let model = LibraryViewModel(initialPath: initialPath)
        self.model = model

        let rootView = RootView(model: model)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1500, height: 940),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "Transcript Viewer"
        window.titlebarAppearsTransparent = true
        window.toolbarStyle = .unified
        window.contentView = NSHostingView(rootView: rootView)
        window.center()
        window.makeKeyAndOrderFront(nil)
        self.window = window

        installMenu()
        model.loadStartupLibrary(initialPath: initialPath)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    private func installMenu() {
        let mainMenu = NSMenu()

        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit Transcript Viewer", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        mainMenu.addItem(appItem)

        let fileItem = NSMenuItem()
        let fileMenu = NSMenu(title: "File")
        fileMenu.addItem(withTitle: "Open Library...", action: #selector(openLibrary), keyEquivalent: "o")
        fileMenu.addItem(withTitle: "Reload Library", action: #selector(reloadLibrary), keyEquivalent: "r")
        fileMenu.addItem(.separator())
        fileMenu.addItem(withTitle: "Reveal Source in Finder", action: #selector(revealSource), keyEquivalent: "")
        fileItem.submenu = fileMenu
        mainMenu.addItem(fileItem)

        let reviewItem = NSMenuItem()
        let reviewMenu = NSMenu(title: "Review")
        reviewMenu.addItem(withTitle: "Start AI Review", action: #selector(startAIReview), keyEquivalent: "b")
        reviewMenu.addItem(withTitle: "Show AI Plan", action: #selector(showAIPlan), keyEquivalent: "a")
        reviewMenu.addItem(withTitle: "Previous AI Pick", action: #selector(previousAIPick), keyEquivalent: "\u{F700}")
        reviewMenu.addItem(withTitle: "Next AI Pick", action: #selector(nextAIPick), keyEquivalent: "\u{F701}")
        reviewMenu.addItem(.separator())
        reviewMenu.addItem(withTitle: "Play/Pause", action: #selector(togglePlayback), keyEquivalent: " ")
        reviewItem.submenu = reviewMenu
        mainMenu.addItem(reviewItem)

        NSApp.mainMenu = mainMenu
    }

    @objc func openLibrary() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Open Library"
        panel.message = "Choose a WhisperX _ai_library folder containing manifest.csv."
        if panel.runModal() == .OK, let url = panel.url {
            Task { await model?.loadLibrary(url) }
        }
    }

    @objc func reloadLibrary() {
        model?.reload()
    }

    @objc func revealSource() {
        model?.revealSourceInFinder()
    }

    @objc func nextAIPick() {
        model?.focusNextAIPick()
    }

    @objc func previousAIPick() {
        model?.focusPreviousAIPick()
    }

    @objc func startAIReview() {
        model?.startAIAssistedReview()
    }

    @objc func showAIPlan() {
        model?.inspectorMode = .aiPlan
    }

    @objc func togglePlayback() {
        model?.togglePlayback()
    }
}
