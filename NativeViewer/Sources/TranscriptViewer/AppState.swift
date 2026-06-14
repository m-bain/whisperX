import AVFoundation
import AppKit
import Foundation
import Observation
import TranscriptViewerCore

@MainActor
@Observable
final class LibraryViewModel {
    enum SelectFilter: String, CaseIterable, Identifiable {
        case all
        case good
        case maybe
        case weak
        case unusable

        var id: String { rawValue }

        var title: String {
            switch self {
            case .all: "All"
            case .good: "Good"
            case .maybe: "Maybe"
            case .weak: "Weak"
            case .unusable: "Unusable"
            }
        }
    }

    enum SelectSort: String, CaseIterable, Identifiable {
        case hook
        case recent
        case timeline
        case status

        var id: String { rawValue }

        var title: String {
            switch self {
            case .hook: "Hook"
            case .recent: "Recent"
            case .timeline: "Timeline"
            case .status: "Status"
            }
        }
    }

    enum SegmentScope: String, CaseIterable, Identifiable {
        case all
        case unreviewed
        case selected
        case good
        case maybe
        case weak

        var id: String { rawValue }

        var title: String {
            switch self {
            case .all: "All"
            case .unreviewed: "Unreviewed"
            case .selected: "Selected"
            case .good: "Good"
            case .maybe: "Maybe"
            case .weak: "Weak"
            }
        }

        var systemImage: String {
            switch self {
            case .all: "text.line.first.and.arrowtriangle.forward"
            case .unreviewed: "circle.dotted"
            case .selected: "pin"
            case .good: "checkmark.circle"
            case .maybe: "questionmark.circle"
            case .weak: "xmark.circle"
            }
        }
    }

    @ObservationIgnored private let store = LibraryStore()
    @ObservationIgnored private let defaults = UserDefaults.standard
    @ObservationIgnored var player = AVPlayer()
    @ObservationIgnored private var currentPlayerURL: URL?
    @ObservationIgnored private let lastLibraryKey = "TranscriptViewer.lastLibraryPath"
    @ObservationIgnored private let recentLibrariesKey = "TranscriptViewer.recentLibraries"
    @ObservationIgnored private var boundaryObserver: Any?

    var libraryURL: URL?
    var recentLibraries: [String] = []
    var files: [TranscriptFile] = []
    var segments: [TranscriptSegment] = []
    var selects: [SelectMoment] = []
    var selectedFileID: String?
    var selectedSegmentID: String?
    var fileSearchText = ""
    var searchText = ""
    var segmentScope: SegmentScope = .all
    var isLoading = false
    var statusMessage = "Open a WhisperX _ai_library folder."

    var draftStatus: SelectStatus = .selected
    var draftHookStrength: Int = 3
    var draftTheme = ""
    var draftTags = ""
    var draftNotes = ""
    var draftStart = ""
    var draftEnd = ""
    var autoAdvanceAfterMark = true
    var showSelectsShelf = true
    var selectFilter: SelectFilter = .all
    var selectSort: SelectSort = .hook
    var minimumHookStrength = 0
    var selectSearchText = ""

    init(initialPath: String?) {
        recentLibraries = defaults.stringArray(forKey: recentLibrariesKey) ?? []
        if let initialPath, !initialPath.isEmpty {
            Task { await loadLibrary(URL(fileURLWithPath: initialPath)) }
        } else if let lastPath = defaults.string(forKey: lastLibraryKey), !lastPath.isEmpty {
            Task { await loadLibrary(URL(fileURLWithPath: lastPath)) }
        }
    }

    var selectedFile: TranscriptFile? {
        guard let selectedFileID else { return nil }
        return files.first { $0.id == selectedFileID }
    }

    var selectedSegment: TranscriptSegment? {
        guard let selectedSegmentID else { return filteredSegments.first }
        return segments.first { $0.id == selectedSegmentID }
    }

    var filteredFiles: [TranscriptFile] {
        let query = fileSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return files }
        return files.filter {
            $0.relativePath.localizedCaseInsensitiveContains(query)
                || $0.status.localizedCaseInsensitiveContains(query)
                || ($0.language?.localizedCaseInsensitiveContains(query) ?? false)
        }
    }

    var doneFileCount: Int {
        files.filter { $0.status == "done" }.count
    }

    var pendingFileCount: Int {
        files.filter { $0.status == "pending" }.count
    }

    var reviewedCount: Int {
        selects.count
    }

    var unreviewedCount: Int {
        max(segments.count - selects.count, 0)
    }

    var progressFraction: Double {
        guard !segments.isEmpty else { return 0 }
        return min(Double(reviewedCount) / Double(segments.count), 1)
    }

    var selectedSegmentIndex: Int? {
        guard let selectedSegmentID else { return nil }
        return filteredSegments.firstIndex { $0.id == selectedSegmentID }
    }

    var reviewPositionText: String {
        guard let selectedSegmentIndex else { return "\(filteredSegments.count) moments" }
        return "\(selectedSegmentIndex + 1) of \(filteredSegments.count)"
    }

    var selectedDurationText: String {
        guard let selectedSegment else { return "No segment selected" }
        return "\(formatTime(selectedSegment.start)) - \(formatTime(selectedSegment.end))"
    }

    var selectedSourcePath: String {
        selectedSegment?.sourceURL.path ?? ""
    }

    var hasSelectedSelect: Bool {
        guard let selectedSegment else { return false }
        return select(for: selectedSegment) != nil
    }

    var filteredSegments: [TranscriptSegment] {
        let byFile: [TranscriptSegment]
        if let selectedFileID {
            byFile = segments.filter { $0.fileID == selectedFileID }
        } else {
            byFile = segments
        }

        let byScope = byFile.filter { segment in
            switch segmentScope {
            case .all:
                true
            case .unreviewed:
                select(for: segment) == nil
            case .selected:
                select(for: segment) != nil
            case .good:
                select(for: segment)?.status == .good
            case .maybe:
                select(for: segment)?.status == .maybe
            case .weak:
                {
                    guard let status = select(for: segment)?.status else { return false }
                    return status == .weak || status == .unusable
                }()
            }
        }

        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return byScope }
        return byScope.filter { segment in
            segment.text.localizedCaseInsensitiveContains(query)
                || segment.relativePath.localizedCaseInsensitiveContains(query)
                || (segment.speaker?.localizedCaseInsensitiveContains(query) ?? false)
        }
    }

    var visibleSelects: [SelectMoment] {
        let query = selectSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return sortedSelects.filter { select in
            let matchesFilter: Bool
            switch selectFilter {
            case .all:
                matchesFilter = true
            case .good:
                matchesFilter = select.status == .good
            case .maybe:
                matchesFilter = select.status == .maybe
            case .weak:
                matchesFilter = select.status == .weak
            case .unusable:
                matchesFilter = select.status == .unusable
            }
            let hook = select.hookStrength ?? 0
            let matchesHook = minimumHookStrength == 0 || hook >= minimumHookStrength
            let matchesQuery = query.isEmpty
                || select.text.localizedCaseInsensitiveContains(query)
                || select.relativePath.localizedCaseInsensitiveContains(query)
                || select.theme.localizedCaseInsensitiveContains(query)
                || select.tags.joined(separator: " ").localizedCaseInsensitiveContains(query)
                || (select.speaker?.localizedCaseInsensitiveContains(query) ?? false)
            return matchesFilter && matchesHook && matchesQuery
        }
    }

    private var sortedSelects: [SelectMoment] {
        selects.sorted { lhs, rhs in
            switch selectSort {
            case .hook:
                return sortByHook(lhs, rhs)
            case .recent:
                return lhs.updatedAt > rhs.updatedAt
            case .timeline:
                if lhs.relativePath == rhs.relativePath {
                    return lhs.start < rhs.start
                }
                return lhs.relativePath.localizedStandardCompare(rhs.relativePath) == .orderedAscending
            case .status:
                let lhsRank = statusRank(lhs.status)
                let rhsRank = statusRank(rhs.status)
                if lhsRank != rhsRank {
                    return lhsRank < rhsRank
                }
                return sortByHook(lhs, rhs)
            }
        }
    }

    private func sortByHook(_ lhs: SelectMoment, _ rhs: SelectMoment) -> Bool {
        let lhsHook = lhs.hookStrength ?? 0
        let rhsHook = rhs.hookStrength ?? 0
        if lhsHook != rhsHook {
            return lhsHook > rhsHook
        }
        if lhs.relativePath == rhs.relativePath {
            return lhs.start < rhs.start
        }
        return lhs.updatedAt > rhs.updatedAt
    }

    private func statusRank(_ status: SelectStatus) -> Int {
        switch status {
        case .good: 0
        case .maybe: 1
        case .selected: 2
        case .weak: 3
        case .unusable: 4
        }
    }

    var selectFilterSummary: String {
        let shown = visibleSelects.count
        let shownLabel = shown == 1 ? "1 select" : "\(shown) selects"
        if shown == selects.count {
            return shownLabel
        }
        let totalLabel = selects.count == 1 ? "1 select" : "\(selects.count) selects"
        return "\(shownLabel) of \(totalLabel)"
    }

    var hasSelectFilters: Bool {
        selectFilter != .all
            || selectSort != .hook
            || minimumHookStrength > 0
            || !selectSearchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func resetSelectFilters() {
        selectFilter = .all
        selectSort = .hook
        minimumHookStrength = 0
        selectSearchText = ""
    }

    func count(for filter: SelectFilter) -> Int {
        switch filter {
        case .all:
            selects.count
        case .good:
            selects.filter { $0.status == .good }.count
        case .maybe:
            selects.filter { $0.status == .maybe }.count
        case .weak:
            selects.filter { $0.status == .weak }.count
        case .unusable:
            selects.filter { $0.status == .unusable }.count
        }
    }

    func loadLibrary(_ url: URL) async {
        isLoading = true
        defer { isLoading = false }
        do {
            let snapshot = try store.load(libraryURL: url)
            libraryURL = snapshot.libraryURL
            files = snapshot.files
            segments = snapshot.segments
            selects = snapshot.selects
            selectedFileID = files.first(where: { $0.segmentCount > 0 })?.id ?? files.first?.id
            selectedSegmentID = filteredSegments.first?.id
            statusMessage = "\(files.count) files, \(segments.count) transcript segments, \(selects.count) selects"
            rememberLibrary(snapshot.libraryURL)
            if let segment = selectedSegment {
                focus(segment, autoplay: false)
            } else {
                clearDraft()
            }
        } catch {
            statusMessage = error.localizedDescription
            clearLibrary()
        }
    }

    func reload() {
        guard let libraryURL else { return }
        Task { await loadLibrary(libraryURL) }
    }

    func choose(file: TranscriptFile?) {
        selectedFileID = file?.id
        selectedSegmentID = filteredSegments.first?.id
        if let segment = selectedSegment {
            focus(segment, autoplay: false)
        } else {
            clearDraft()
        }
    }

    func clearFileSelection() {
        choose(file: nil)
    }

    func startUnreviewedReview() {
        selectedFileID = nil
        setScope(.unreviewed)
        statusMessage = "Reviewing unreviewed moments"
    }

    func focus(_ segment: TranscriptSegment, autoplay: Bool) {
        if selectedFileID != nil {
            selectedFileID = segment.fileID
        }
        selectedSegmentID = segment.id
        preparePlayer(for: segment)
        player.seek(to: CMTime(seconds: segment.start, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        installBoundaryObserver(end: segment.end)
        if autoplay {
            player.play()
        }
        loadDraft(for: segment)
    }

    func focus(_ select: SelectMoment, autoplay: Bool) {
        if let segment = segments.first(where: { $0.id == select.segmentID }) {
            selectedFileID = nil
            selectedSegmentID = segment.id
            focus(segment, autoplay: autoplay)
        } else {
            statusMessage = "Select source segment is no longer in this library"
        }
    }

    func focusSelectedWithoutAutoplay() {
        guard let segment = selectedSegment else { return }
        focus(segment, autoplay: false)
    }

    func focusNext(autoplay: Bool = true) {
        moveSelection(offset: 1, autoplay: autoplay)
    }

    func focusPrevious(autoplay: Bool = true) {
        moveSelection(offset: -1, autoplay: autoplay)
    }

    func moveSelection(offset: Int, autoplay: Bool) {
        guard !filteredSegments.isEmpty else { return }
        let currentIndex = selectedSegmentIndex ?? 0
        let nextIndex = min(max(currentIndex + offset, 0), filteredSegments.count - 1)
        focus(filteredSegments[nextIndex], autoplay: autoplay)
    }

    func setScope(_ scope: SegmentScope) {
        segmentScope = scope
        if let selectedSegmentID, filteredSegments.contains(where: { $0.id == selectedSegmentID }) {
            focusSelectedWithoutAutoplay()
        } else {
            selectedSegmentID = filteredSegments.first?.id
            focusSelectedWithoutAutoplay()
        }
    }

    func nudgePlayback(seconds: Double) {
        let next = max(0, player.currentTime().seconds + seconds)
        player.seek(to: CMTime(seconds: next, preferredTimescale: 600))
    }

    func togglePlayback() {
        if player.timeControlStatus == .playing {
            player.pause()
        } else {
            player.play()
        }
    }

    func mark(status: SelectStatus, hookStrength: Int? = nil, advance: Bool = false) {
        let nextSegment = advance ? nextSegmentAfterCurrent() : nil
        draftStatus = status
        if let hookStrength {
            draftHookStrength = hookStrength
        }
        if saveDraft(), advance {
            if let nextSegment {
                focus(nextSegment, autoplay: true)
            } else {
                settleSelectionAfterFilterChange()
            }
        }
    }

    func setHookStrength(_ value: Int) {
        draftHookStrength = min(max(value, 1), 5)
        if hasSelectedSelect {
            saveDraft()
        } else {
            statusMessage = "Hook strength set to \(draftHookStrength)"
        }
    }

    @discardableResult
    func saveDraft(status overrideStatus: SelectStatus? = nil, hookStrength overrideHook: Int? = nil) -> Bool {
        guard let libraryURL, let segment = selectedSegment else { return false }
        let existing = select(for: segment)
        let now = Date()
        let start = Double(draftStart) ?? segment.start
        let end = max(start, Double(draftEnd) ?? segment.end)
        let tags = draftTags
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let moment = SelectMoment(
            id: existing?.id ?? "select::\(segment.id)",
            segmentID: segment.id,
            fileID: segment.fileID,
            sourceURL: segment.sourceURL,
            relativePath: segment.relativePath,
            start: start,
            end: end,
            speaker: segment.speaker,
            text: segment.text,
            status: overrideStatus ?? draftStatus,
            hookStrength: overrideHook ?? draftHookStrength,
            theme: draftTheme,
            tags: tags,
            notes: draftNotes,
            createdAt: existing?.createdAt ?? now,
            updatedAt: now
        )
        if let index = selects.firstIndex(where: { $0.segmentID == segment.id }) {
            selects[index] = moment
        } else {
            selects.append(moment)
        }
        do {
            try store.save(selects: selects, libraryURL: libraryURL)
            loadDraft(for: segment)
            statusMessage = "Saved \(moment.status.title.lowercased()) select at \(formatTime(moment.start))"
            return true
        } catch {
            statusMessage = error.localizedDescription
            return false
        }
    }

    func deleteSelectedSelect() {
        guard let libraryURL, let segment = selectedSegment else { return }
        selects.removeAll { $0.segmentID == segment.id }
        do {
            try store.save(selects: selects, libraryURL: libraryURL)
            loadDraft(for: segment)
            statusMessage = "Removed select"
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func exportCSV() {
        guard let libraryURL else { return }
        do {
            let url = try store.export(selects: selects, libraryURL: libraryURL)
            statusMessage = "Exported \(selects.count) selects to \(url.lastPathComponent)"
        } catch {
            statusMessage = error.localizedDescription
        }
    }

    func copySelectedSelectCSV() {
        guard let segment = selectedSegment, let select = select(for: segment) else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(CSV.encode(rows: [csvRow(for: select)]).trimmingCharacters(in: .newlines), forType: .string)
        statusMessage = "Copied select CSV row"
    }

    func revealSourceInFinder() {
        guard let segment = selectedSegment else { return }
        NSWorkspace.shared.activateFileViewerSelecting([segment.sourceURL])
    }

    func revealExportInFinder() {
        guard let libraryURL else { return }
        let url = libraryURL.appendingPathComponent(LibraryStore.exportFilename)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func adjustDraftStart(by delta: Double) {
        guard let segment = selectedSegment else { return }
        let current = Double(draftStart) ?? segment.start
        let end = Double(draftEnd) ?? segment.end
        draftStart = String(format: "%.3f", min(max(0, current + delta), end))
    }

    func adjustDraftEnd(by delta: Double) {
        guard let segment = selectedSegment else { return }
        let start = Double(draftStart) ?? segment.start
        let current = Double(draftEnd) ?? segment.end
        draftEnd = String(format: "%.3f", max(start, current + delta))
    }

    func select(for segment: TranscriptSegment) -> SelectMoment? {
        selects.first { $0.segmentID == segment.id }
    }

    func selectCount(for file: TranscriptFile) -> Int {
        selects.filter { $0.fileID == file.id }.count
    }

    func count(for scope: SegmentScope) -> Int {
        switch scope {
        case .all:
            segments.count
        case .unreviewed:
            unreviewedCount
        case .selected:
            selects.count
        case .good:
            selects.filter { $0.status == .good }.count
        case .maybe:
            selects.filter { $0.status == .maybe }.count
        case .weak:
            selects.filter { $0.status == .weak || $0.status == .unusable }.count
        }
    }

    func formatTime(_ seconds: Double) -> String {
        let totalMilliseconds = Int((seconds * 1000).rounded())
        let minutes = totalMilliseconds / 60_000
        let seconds = (totalMilliseconds % 60_000) / 1000
        let tenths = (totalMilliseconds % 1000) / 100
        return "\(minutes):\(String(format: "%02d", seconds)).\(tenths)"
    }

    private func preparePlayer(for segment: TranscriptSegment) {
        guard currentPlayerURL != segment.sourceURL else { return }
        currentPlayerURL = segment.sourceURL
        player.replaceCurrentItem(with: AVPlayerItem(url: segment.sourceURL))
    }

    private func csvRow(for select: SelectMoment) -> [String] {
        [
            select.sourceURL.path,
            select.relativePath,
            String(format: "%.3f", select.start),
            String(format: "%.3f", select.end),
            select.theme,
            select.hookStrength.map(String.init) ?? "",
            select.status.rawValue,
            select.tags.joined(separator: ","),
            select.speaker ?? "",
            select.text,
            select.notes,
            select.segmentID
        ]
    }

    private func installBoundaryObserver(end: Double) {
        if let boundaryObserver {
            player.removeTimeObserver(boundaryObserver)
            self.boundaryObserver = nil
        }
        let endTime = CMTime(seconds: end, preferredTimescale: 600)
        boundaryObserver = player.addBoundaryTimeObserver(forTimes: [NSValue(time: endTime)], queue: .main) { [weak self] in
            Task { @MainActor in
                self?.player.pause()
            }
        }
    }

    private func nextSegmentAfterCurrent() -> TranscriptSegment? {
        guard !filteredSegments.isEmpty else { return nil }
        let currentIndex = selectedSegmentIndex ?? 0
        if currentIndex + 1 < filteredSegments.count {
            return filteredSegments[currentIndex + 1]
        }
        if currentIndex > 0 {
            return filteredSegments[currentIndex - 1]
        }
        return nil
    }

    private func settleSelectionAfterFilterChange() {
        if let selectedSegmentID, filteredSegments.contains(where: { $0.id == selectedSegmentID }) {
            focusSelectedWithoutAutoplay()
            return
        }
        selectedSegmentID = filteredSegments.first?.id
        if let segment = selectedSegment {
            focus(segment, autoplay: false)
        } else {
            clearDraft()
            player.pause()
            statusMessage = "No more moments in this queue"
        }
    }

    private func loadDraft(for segment: TranscriptSegment) {
        if let select = select(for: segment) {
            draftStatus = select.status
            draftHookStrength = select.hookStrength ?? 3
            draftTheme = select.theme
            draftTags = select.tags.joined(separator: ", ")
            draftNotes = select.notes
            draftStart = String(format: "%.3f", select.start)
            draftEnd = String(format: "%.3f", select.end)
        } else {
            draftStatus = .selected
            draftHookStrength = 3
            draftTheme = ""
            draftTags = ""
            draftNotes = ""
            draftStart = String(format: "%.3f", segment.start)
            draftEnd = String(format: "%.3f", segment.end)
        }
    }

    private func clearDraft() {
        draftStatus = .selected
        draftHookStrength = 3
        draftTheme = ""
        draftTags = ""
        draftNotes = ""
        draftStart = ""
        draftEnd = ""
    }

    private func rememberLibrary(_ url: URL) {
        let path = url.path
        defaults.set(path, forKey: lastLibraryKey)
        recentLibraries.removeAll { $0 == path }
        recentLibraries.insert(path, at: 0)
        recentLibraries = Array(recentLibraries.prefix(6))
        defaults.set(recentLibraries, forKey: recentLibrariesKey)
    }

    private func clearLibrary() {
        libraryURL = nil
        files = []
        segments = []
        selects = []
        selectedFileID = nil
        selectedSegmentID = nil
        player.pause()
        if let boundaryObserver {
            player.removeTimeObserver(boundaryObserver)
            self.boundaryObserver = nil
        }
        player.replaceCurrentItem(with: nil)
        currentPlayerURL = nil
        clearDraft()
    }
}
