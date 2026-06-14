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
        fileMenu.addItem(withTitle: "Export Selects CSV", action: #selector(exportCSV), keyEquivalent: "e")
        fileMenu.addItem(withTitle: "Reveal Export in Finder", action: #selector(revealExport), keyEquivalent: "")
        fileMenu.addItem(withTitle: "Reveal Source in Finder", action: #selector(revealSource), keyEquivalent: "")
        fileItem.submenu = fileMenu
        mainMenu.addItem(fileItem)

        let reviewItem = NSMenuItem()
        let reviewMenu = NSMenu(title: "Review")
        reviewMenu.addItem(withTitle: "Start Unreviewed Review", action: #selector(startUnreviewedReview), keyEquivalent: "b")
        reviewMenu.addItem(withTitle: "Toggle Selects Shelf", action: #selector(toggleSelectsShelf), keyEquivalent: "l")
        reviewMenu.addItem(.separator())
        reviewMenu.addItem(withTitle: "Previous Moment", action: #selector(previousMoment), keyEquivalent: "\u{F700}")
        reviewMenu.addItem(withTitle: "Next Moment", action: #selector(nextMoment), keyEquivalent: "\u{F701}")
        reviewMenu.addItem(withTitle: "Play/Pause", action: #selector(togglePlayback), keyEquivalent: " ")
        reviewMenu.addItem(.separator())
        reviewMenu.addItem(withTitle: "Mark Good", action: #selector(markGood), keyEquivalent: "g")
        reviewMenu.addItem(withTitle: "Mark Maybe", action: #selector(markMaybe), keyEquivalent: "m")
        reviewMenu.addItem(withTitle: "Mark Weak", action: #selector(markWeak), keyEquivalent: "x")
        reviewMenu.addItem(withTitle: "Mark Unusable", action: #selector(markUnusable), keyEquivalent: "u")
        reviewMenu.addItem(.separator())
        reviewMenu.addItem(withTitle: "Copy Select CSV Row", action: #selector(copySelectCSV), keyEquivalent: "c")
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

    @objc func exportCSV() {
        model?.exportCSV()
    }

    @objc func revealExport() {
        model?.revealExportInFinder()
    }

    @objc func revealSource() {
        model?.revealSourceInFinder()
    }

    @objc func nextMoment() {
        model?.focusNext()
    }

    @objc func startUnreviewedReview() {
        model?.startUnreviewedReview()
    }

    @objc func toggleSelectsShelf() {
        guard let model else { return }
        model.showSelectsShelf.toggle()
    }

    @objc func copySelectCSV() {
        model?.copySelectedSelectCSV()
    }

    @objc func previousMoment() {
        model?.focusPrevious()
    }

    @objc func togglePlayback() {
        model?.togglePlayback()
    }

    @objc func markGood() {
        guard let model else { return }
        model.mark(status: .good, hookStrength: 5, advance: model.autoAdvanceAfterMark)
    }

    @objc func markMaybe() {
        guard let model else { return }
        model.mark(status: .maybe, advance: model.autoAdvanceAfterMark)
    }

    @objc func markWeak() {
        guard let model else { return }
        model.mark(status: .weak, advance: model.autoAdvanceAfterMark)
    }

    @objc func markUnusable() {
        guard let model else { return }
        model.mark(status: .unusable, hookStrength: 1, advance: model.autoAdvanceAfterMark)
    }
}
