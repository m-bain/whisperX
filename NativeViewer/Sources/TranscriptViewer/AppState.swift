import AVFoundation
import AppKit
import Foundation
import Observation
import TranscriptViewerCore

@MainActor
@Observable
final class LibraryViewModel {
    private static let allThemesFilter = "All Themes"
    private static let allQualitiesFilter = "All Qualities"

    enum SegmentScope: String, CaseIterable, Identifiable {
        case all
        case aiPicks
        case highHooks
        case noAIPick

        var id: String { rawValue }

        var title: String {
            switch self {
            case .all: "All"
            case .aiPicks: "AI Picks"
            case .highHooks: "High Hooks"
            case .noAIPick: "No AI Pick"
            }
        }

        var systemImage: String {
            switch self {
            case .all: "text.line.first.and.arrowtriangle.forward"
            case .aiPicks: "sparkles"
            case .highHooks: "flame"
            case .noAIPick: "line.3.horizontal.decrease.circle"
            }
        }
    }

    enum InspectorMode: String, CaseIterable, Identifiable {
        case moment
        case aiPlan

        var id: String { rawValue }

        var title: String {
            switch self {
            case .moment: "AI Pick"
            case .aiPlan: "AI Plan"
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
    var clipMoments: [ClipMoment] = []
    var analysisArtifacts: [AnalysisArtifact] = []
    var selectedFileID: String?
    var selectedSegmentID: String?
    var selectedClipMomentID: String?
    var selectedAnalysisArtifactID: String?
    var fileSearchText = ""
    var searchText = ""
    var segmentScope: SegmentScope = .all
    var inspectorMode: InspectorMode = .moment
    var isLoading = false
    var statusMessage = "Open a WhisperX _ai_library folder."

    var clipSearchText = ""
    var clipThemeFilter = allThemesFilter
    var clipQualityFilter = allQualitiesFilter

    init(initialPath: String?) {
        recentLibraries = defaults.stringArray(forKey: recentLibrariesKey) ?? []
    }

    func loadStartupLibrary(initialPath: String?) {
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

    var selectedClipMoment: ClipMoment? {
        guard let selectedClipMomentID else { return filteredClipMoments.first }
        return clipMoments.first { $0.id == selectedClipMomentID }
    }

    var currentAIPick: ClipMoment? {
        guard let selectedSegment else {
            return selectedClipMoment
        }
        let matches = matchingClipMoments(for: selectedSegment)
        if let selectedClipMoment, matches.contains(where: { $0.id == selectedClipMoment.id }) {
            return selectedClipMoment
        }
        return matches.first
    }

    var selectedAnalysisArtifact: AnalysisArtifact? {
        guard let selectedAnalysisArtifactID else { return analysisArtifacts.first }
        return analysisArtifacts.first { $0.id == selectedAnalysisArtifactID }
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

    var highPriorityClipMomentCount: Int {
        clipMoments.filter { hookRank($0.hookStrength) >= 5 }.count
    }

    var progressFraction: Double {
        guard !files.isEmpty else { return 0 }
        return min(Double(doneFileCount) / Double(files.count), 1)
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
            case .aiPicks:
                !matchingClipMoments(for: segment).isEmpty
            case .highHooks:
                matchingClipMoments(for: segment).contains { hookRank($0.hookStrength) >= 5 }
            case .noAIPick:
                matchingClipMoments(for: segment).isEmpty
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

    var clipThemes: [String] {
        let themes = Set(clipMoments.map(\.theme).filter { !$0.isEmpty })
        return [Self.allThemesFilter] + themes.sorted {
            $0.localizedStandardCompare($1) == .orderedAscending
        }
    }

    var clipQualities: [String] {
        let qualities = Set(clipMoments.map(\.quality).filter { !$0.isEmpty })
        return [Self.allQualitiesFilter] + qualities.sorted {
            $0.localizedStandardCompare($1) == .orderedAscending
        }
    }

    var filteredClipMoments: [ClipMoment] {
        let query = clipSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return clipMoments
            .filter { moment in
                let matchesTheme = clipThemeFilter == Self.allThemesFilter || moment.theme == clipThemeFilter
                let matchesQuality = clipQualityFilter == Self.allQualitiesFilter || moment.quality == clipQualityFilter
                let matchesQuery = query.isEmpty
                    || moment.text.localizedCaseInsensitiveContains(query)
                    || moment.relativePath.localizedCaseInsensitiveContains(query)
                    || moment.theme.localizedCaseInsensitiveContains(query)
                    || moment.quality.localizedCaseInsensitiveContains(query)
                    || (moment.speaker?.localizedCaseInsensitiveContains(query) ?? false)
                return matchesTheme && matchesQuality && matchesQuery
            }
            .sorted { lhs, rhs in
                let lhsRank = hookRank(lhs.hookStrength)
                let rhsRank = hookRank(rhs.hookStrength)
                if lhsRank != rhsRank {
                    return lhsRank > rhsRank
                }
                if lhs.relativePath == rhs.relativePath {
                    return lhs.start < rhs.start
                }
                return lhs.relativePath.localizedStandardCompare(rhs.relativePath) == .orderedAscending
            }
    }

    var clipFilterSummary: String {
        let shown = filteredClipMoments.count
        let shownLabel = shown == 1 ? "1 AI pick" : "\(shown) AI picks"
        if shown == clipMoments.count {
            return shownLabel
        }
        return "\(shownLabel) of \(clipMoments.count)"
    }

    var analysisSearchPlaceholder: String {
        "\(analysisArtifacts.count) generated notes"
    }

    func loadLibrary(_ url: URL) async {
        isLoading = true
        defer { isLoading = false }
        do {
            let snapshot = try store.load(libraryURL: url)
            libraryURL = snapshot.libraryURL
            files = snapshot.files
            segments = snapshot.segments
            clipMoments = snapshot.clipMoments
            analysisArtifacts = snapshot.analysisArtifacts
            selectedAnalysisArtifactID = analysisArtifacts.first?.id
            if !clipThemes.contains(clipThemeFilter) {
                clipThemeFilter = Self.allThemesFilter
            }
            if !clipQualities.contains(clipQualityFilter) {
                clipQualityFilter = Self.allQualitiesFilter
            }
            statusMessage = "\(files.count) files, \(segments.count) transcript segments, \(clipMoments.count) AI picks"
            rememberLibrary(snapshot.libraryURL)
            if let firstPick = filteredClipMoments.first(where: { bestSegment(for: $0) != nil }) {
                selectedFileID = nil
                segmentScope = .aiPicks
                focus(firstPick, autoplay: false)
            } else if let segment = segments.first {
                selectedFileID = segment.fileID
                selectedSegmentID = segment.id
                focus(segment, autoplay: false)
            } else {
                selectedFileID = files.first?.id
                selectedSegmentID = nil
                player.pause()
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
            player.pause()
        }
    }

    func clearFileSelection() {
        choose(file: nil)
    }

    func startAIAssistedReview() {
        selectedFileID = nil
        inspectorMode = .aiPlan
        segmentScope = .all
        guard let firstPick = filteredClipMoments.first else {
            statusMessage = "No AI picks available"
            return
        }
        focus(firstPick, autoplay: false)
        statusMessage = "Reviewing AI-ranked picks"
    }

    func focusHighHookAIPick() {
        selectedFileID = nil
        inspectorMode = .aiPlan
        segmentScope = .all
        guard let pick = filteredClipMoments.first(where: { hookRank($0.hookStrength) >= 5 })
            ?? clipMoments.first(where: { hookRank($0.hookStrength) >= 5 })
            ?? filteredClipMoments.first
        else {
            statusMessage = "No AI picks available"
            return
        }
        focus(pick, autoplay: true)
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
        let matches = matchingClipMoments(for: segment)
        if let selectedClipMomentID, matches.contains(where: { $0.id == selectedClipMomentID }) {
            return
        }
        selectedClipMomentID = matches.first?.id
    }

    func focus(_ clipMoment: ClipMoment, autoplay: Bool) {
        selectedClipMomentID = clipMoment.id
        inspectorMode = .aiPlan
        guard let segment = bestSegment(for: clipMoment) else {
            statusMessage = "No matching transcript segment for \(clipMoment.relativePath)"
            return
        }
        selectedFileID = nil
        focus(segment, autoplay: autoplay)
        statusMessage = "Focused AI pick at \(formatTime(clipMoment.start))"
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

    func focusNextAIPick(autoplay: Bool = true) {
        moveAIPick(offset: 1, autoplay: autoplay)
    }

    func focusPreviousAIPick(autoplay: Bool = true) {
        moveAIPick(offset: -1, autoplay: autoplay)
    }

    func moveSelection(offset: Int, autoplay: Bool) {
        guard !filteredSegments.isEmpty else { return }
        let currentIndex = selectedSegmentIndex ?? 0
        let nextIndex = min(max(currentIndex + offset, 0), filteredSegments.count - 1)
        focus(filteredSegments[nextIndex], autoplay: autoplay)
    }

    func moveAIPick(offset: Int, autoplay: Bool) {
        let picks = filteredClipMoments
        guard !picks.isEmpty else {
            statusMessage = "No AI picks available"
            return
        }
        let currentIndex = selectedClipMomentID.flatMap { id in picks.firstIndex { $0.id == id } } ?? 0
        let nextIndex = min(max(currentIndex + offset, 0), picks.count - 1)
        focus(picks[nextIndex], autoplay: autoplay)
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

    func revealSourceInFinder() {
        guard let segment = selectedSegment else { return }
        NSWorkspace.shared.activateFileViewerSelecting([segment.sourceURL])
    }

    func count(for scope: SegmentScope) -> Int {
        switch scope {
        case .all:
            segments.count
        case .aiPicks:
            segments.filter { !matchingClipMoments(for: $0).isEmpty }.count
        case .highHooks:
            segments.filter { segment in
                matchingClipMoments(for: segment).contains { hookRank($0.hookStrength) >= 5 }
            }.count
        case .noAIPick:
            segments.filter { matchingClipMoments(for: $0).isEmpty }.count
        }
    }

    func matchingClipMoments(for segment: TranscriptSegment) -> [ClipMoment] {
        clipMoments.filter { clipMoment in
            clipMoment.relativePath == segment.relativePath && overlaps(clipMoment, segment)
        }
        .sorted { hookRank($0.hookStrength) > hookRank($1.hookStrength) }
    }

    func hookRank(_ hookStrength: String) -> Int {
        switch hookStrength.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "high", "strong", "5":
            5
        case "medium", "med", "4":
            4
        case "low", "3":
            3
        case "weak", "2":
            2
        case "bad", "1":
            1
        default:
            3
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

    private func bestSegment(for clipMoment: ClipMoment) -> TranscriptSegment? {
        let candidates = segments.filter { $0.relativePath == clipMoment.relativePath }
        if candidates.isEmpty {
            return nil
        }
        let midpoint = (clipMoment.start + clipMoment.end) / 2
        return candidates.min { lhs, rhs in
            let lhsOverlap = overlapDuration(clipMoment, lhs)
            let rhsOverlap = overlapDuration(clipMoment, rhs)
            if lhsOverlap != rhsOverlap {
                return lhsOverlap > rhsOverlap
            }
            let lhsDistance = abs(((lhs.start + lhs.end) / 2) - midpoint)
            let rhsDistance = abs(((rhs.start + rhs.end) / 2) - midpoint)
            return lhsDistance < rhsDistance
        }
    }

    private func overlaps(_ clipMoment: ClipMoment, _ segment: TranscriptSegment) -> Bool {
        overlapDuration(clipMoment, segment) > 0
    }

    private func overlapDuration(_ clipMoment: ClipMoment, _ segment: TranscriptSegment) -> Double {
        max(0, min(clipMoment.end, segment.end) - max(clipMoment.start, segment.start))
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
            player.pause()
            statusMessage = "No more moments in this queue"
        }
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
        clipMoments = []
        analysisArtifacts = []
        selectedFileID = nil
        selectedSegmentID = nil
        selectedClipMomentID = nil
        selectedAnalysisArtifactID = nil
        player.pause()
        if let boundaryObserver {
            player.removeTimeObserver(boundaryObserver)
            self.boundaryObserver = nil
        }
        player.replaceCurrentItem(with: nil)
        currentPlayerURL = nil
    }
}
