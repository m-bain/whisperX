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
        case people
        case noAIPick

        var id: String { rawValue }

        var title: String {
            switch self {
            case .all: "All"
            case .aiPicks: "AI Picks"
            case .highHooks: "High Hooks"
            case .people: "People"
            case .noAIPick: "No AI Pick"
            }
        }

        var systemImage: String {
            switch self {
            case .all: "text.line.first.and.arrowtriangle.forward"
            case .aiPicks: "sparkles"
            case .highHooks: "flame"
            case .people: "person.crop.rectangle.stack"
            case .noAIPick: "line.3.horizontal.decrease.circle"
            }
        }
    }

    enum InspectorMode: String, CaseIterable, Identifiable {
        case moment
        case aiPlan
        case person

        var id: String { rawValue }

        var title: String {
            switch self {
            case .moment: "AI Pick"
            case .aiPlan: "AI Plan"
            case .person: "Person"
            }
        }
    }

    enum LibraryMode: String, CaseIterable, Identifiable {
        case review
        case tags

        var id: String { rawValue }
    }

    struct TagSummary: Identifiable, Hashable {
        var id: String
        var label: String
        var clipCount: Int
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
    var clipTags: [ClipTag] = []
    var analysisArtifacts: [AnalysisArtifact] = []
    var people: [PersonProfile] = []
    var selectedFileID: String?
    var selectedSegmentID: String?
    var selectedClipMomentID: String?
    var selectedAnalysisArtifactID: String?
    var selectedPersonID: String?
    var selectedTagFilter: String?
    var fileSearchText = ""
    var personSearchText = ""
    var tagSearchText = ""
    var searchText = ""
    var segmentScope: SegmentScope = .all
    var inspectorMode: InspectorMode = .moment
    var libraryMode: LibraryMode = .review
    var isLoading = false
    var isScanningPeople = false
    var statusMessage = "Open a WhisperX _ai_library folder."

    var clipSearchText = ""
    var clipThemeFilter = allThemesFilter
    var clipQualityFilter = allQualitiesFilter
    var personDraftName = ""
    var personDraftTags = ""
    var personDraftNotes = ""

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

    var selectedClipTags: ClipTag? {
        if let selectedSegment {
            return clipTags(for: selectedSegment.relativePath)
        }
        if let selectedClipMoment {
            return clipTags(for: selectedClipMoment.relativePath)
        }
        if let selectedFile {
            return clipTags(for: selectedFile.relativePath)
        }
        return nil
    }

    var selectedPerson: PersonProfile? {
        guard let selectedPersonID else { return nil }
        return people.first { $0.id == selectedPersonID }
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
        let byTag = files.filter { matchesSelectedTag(relativePath: $0.relativePath) }
        guard !query.isEmpty else { return byTag }
        return byTag.filter {
            $0.relativePath.localizedCaseInsensitiveContains(query)
                || $0.status.localizedCaseInsensitiveContains(query)
                || ($0.language?.localizedCaseInsensitiveContains(query) ?? false)
                || clipTagsMatch(relativePath: $0.relativePath, query: query)
        }
    }

    var filteredPeople: [PersonProfile] {
        let query = personSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = people.filter { !$0.appearances.isEmpty || !$0.displayName.isEmpty || !$0.tags.isEmpty || !$0.notes.isEmpty }
        guard !query.isEmpty else { return base }
        return base.filter { person in
            person.title.localizedCaseInsensitiveContains(query)
                || person.tags.joined(separator: " ").localizedCaseInsensitiveContains(query)
                || person.notes.localizedCaseInsensitiveContains(query)
                || person.appearances.contains { $0.relativePath.localizedCaseInsensitiveContains(query) }
            }
    }

    var tagSummaries: [TagSummary] {
        var labels: [String: String] = [:]
        var counts: [String: Int] = [:]

        for clipTag in clipTags {
            var clipKeys = Set<String>()
            for tag in clipTag.displayTags {
                let label = tag.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !label.isEmpty else { continue }
                let key = tagKey(label)
                labels[key] = labels[key] ?? label
                clipKeys.insert(key)
            }
            for key in clipKeys {
                counts[key, default: 0] += 1
            }
        }

        return counts.map { key, count in
            TagSummary(id: key, label: labels[key] ?? key, clipCount: count)
        }
        .sorted { lhs, rhs in
            if lhs.clipCount != rhs.clipCount {
                return lhs.clipCount > rhs.clipCount
            }
            return lhs.label.localizedStandardCompare(rhs.label) == .orderedAscending
        }
    }

    var filteredTagSummaries: [TagSummary] {
        let query = tagSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return tagSummaries }
        return tagSummaries.filter { $0.label.localizedCaseInsensitiveContains(query) }
    }

    var selectedTagFilterLabel: String {
        selectedTagFilter ?? "All tags"
    }

    var selectedTagClipTags: [ClipTag] {
        guard selectedTagFilter != nil else { return clipTags.filter { !$0.displayTags.isEmpty } }
        return clipTags.filter { clipTag in
            clipTag.displayTags.contains { tagKey($0) == tagKey(selectedTagFilter ?? "") }
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

        let byPerson: [TranscriptSegment]
        if let selectedPerson {
            byPerson = byFile.filter { selectedPerson.fileIDs.contains($0.fileID) }
        } else {
            byPerson = byFile
        }

        let byTag = byPerson.filter { segment in
            matchesSelectedTag(relativePath: segment.relativePath)
        }

        let byScope = byTag.filter { segment in
            switch segmentScope {
            case .all:
                true
            case .aiPicks:
                !matchingClipMoments(for: segment).isEmpty
            case .highHooks:
                matchingClipMoments(for: segment).contains { hookRank($0.hookStrength) >= 5 }
            case .people:
                people.contains { $0.fileIDs.contains(segment.fileID) }
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
                || clipTagsMatch(relativePath: segment.relativePath, query: query)
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
                let matchesTag = matchesSelectedTag(relativePath: moment.relativePath)
                let matchesQuery = query.isEmpty
                    || moment.text.localizedCaseInsensitiveContains(query)
                    || moment.relativePath.localizedCaseInsensitiveContains(query)
                    || moment.theme.localizedCaseInsensitiveContains(query)
                    || moment.quality.localizedCaseInsensitiveContains(query)
                    || (moment.speaker?.localizedCaseInsensitiveContains(query) ?? false)
                    || clipTagsMatch(relativePath: moment.relativePath, query: query)
                return matchesTheme && matchesQuality && matchesTag && matchesQuery
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
            clipTags = snapshot.clipTags
            analysisArtifacts = snapshot.analysisArtifacts
            people = snapshot.people
            selectedAnalysisArtifactID = analysisArtifacts.first?.id
            selectedPersonID = nil
            resetPersonDrafts()
            if !clipThemes.contains(clipThemeFilter) {
                clipThemeFilter = Self.allThemesFilter
            }
            if !clipQualities.contains(clipQualityFilter) {
                clipQualityFilter = Self.allQualitiesFilter
            }
            if let selectedTagFilter, !tagSummaries.contains(where: { tagKey($0.label) == tagKey(selectedTagFilter) }) {
                self.selectedTagFilter = nil
            }
            statusMessage = "\(files.count) files, \(segments.count) transcript segments, \(clipMoments.count) AI picks, \(clipTags.count) tagged clips, \(people.count) people"
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
        libraryMode = .review
        selectedPersonID = nil
        resetPersonDrafts()
        selectedFileID = file?.id
        selectedSegmentID = filteredSegments.first?.id
        if let segment = selectedSegment {
            focus(segment, autoplay: false)
        } else {
            player.pause()
        }
    }

    func clearFileSelection() {
        selectedTagFilter = nil
        choose(file: nil)
    }

    func choose(person: PersonProfile?) {
        libraryMode = .review
        selectedTagFilter = nil
        selectedPersonID = person?.id
        selectedFileID = nil
        inspectorMode = person == nil ? inspectorMode : .person
        segmentScope = person == nil ? segmentScope : .all
        loadPersonDrafts()
        if let appearance = person?.appearances.first,
           let segment = bestSegment(relativePath: appearance.relativePath, timestamp: appearance.timestamp) {
            focus(segment, autoplay: false)
            player.seek(to: CMTime(seconds: appearance.timestamp, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
        } else {
            selectedSegmentID = filteredSegments.first?.id
            focusSelectedWithoutAutoplay()
        }
    }

    func clearPersonSelection() {
        choose(person: nil)
    }

    func showTagCloud() {
        libraryMode = .tags
        selectedFileID = nil
        selectedPersonID = nil
        resetPersonDrafts()
        settleSelectionAfterFilterChange()
    }

    func selectTag(_ tag: TagSummary) {
        libraryMode = .tags
        selectedTagFilter = tag.label
        fileSearchText = ""
        searchText = ""
        clipSearchText = ""
        selectedFileID = nil
        selectedPersonID = nil
        resetPersonDrafts()
        segmentScope = .all
        settleSelectionAfterFilterChange()
        statusMessage = "Filtering \(tag.clipCount) clips tagged \(tag.label)"
    }

    func clearTagFilter() {
        selectedTagFilter = nil
        settleSelectionAfterFilterChange()
        statusMessage = "Showing all tagged clips"
    }

    func startAIAssistedReview() {
        libraryMode = .review
        selectedFileID = nil
        selectedPersonID = nil
        resetPersonDrafts()
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
        libraryMode = .review
        selectedFileID = nil
        selectedPersonID = nil
        resetPersonDrafts()
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
        let tagFilteredSegments = segments.filter { matchesSelectedTag(relativePath: $0.relativePath) }
        switch scope {
        case .all:
            return tagFilteredSegments.count
        case .aiPicks:
            return tagFilteredSegments.filter { !matchingClipMoments(for: $0).isEmpty }.count
        case .highHooks:
            return tagFilteredSegments.filter { segment in
                matchingClipMoments(for: segment).contains { hookRank($0.hookStrength) >= 5 }
            }.count
        case .people:
            let fileIDs = Set(people.flatMap { $0.fileIDs })
            return tagFilteredSegments.filter { fileIDs.contains($0.fileID) }.count
        case .noAIPick:
            return tagFilteredSegments.filter { matchingClipMoments(for: $0).isEmpty }.count
        }
    }

    func scanPeople() {
        guard let libraryURL else { return }
        let scanFiles = files.filter { $0.status == "done" }
        guard !scanFiles.isEmpty else {
            statusMessage = "No completed videos available for people scan"
            return
        }
        isScanningPeople = true
        statusMessage = "Scanning \(scanFiles.count) videos for faces..."
        Task {
            do {
                let appearances = try await FaceIndexer().scan(files: scanFiles)
                try store.replacePeopleIndex(libraryURL: libraryURL, appearances: appearances)
                let snapshot = try store.load(libraryURL: libraryURL)
                files = snapshot.files
                segments = snapshot.segments
                clipMoments = snapshot.clipMoments
                clipTags = snapshot.clipTags
                analysisArtifacts = snapshot.analysisArtifacts
                people = snapshot.people
                if let selectedPersonID, !people.contains(where: { $0.id == selectedPersonID }) {
                    self.selectedPersonID = nil
                    resetPersonDrafts()
                } else {
                    loadPersonDrafts()
                }
                statusMessage = "Found \(people.count) people across \(appearances.count) face appearances"
            } catch {
                statusMessage = "People scan failed: \(error.localizedDescription)"
            }
            isScanningPeople = false
        }
    }

    func saveSelectedPersonTags() {
        guard let libraryURL, let selectedPersonID, let index = people.firstIndex(where: { $0.id == selectedPersonID }) else {
            return
        }
        let updated = PersonProfile(
            id: people[index].id,
            displayName: personDraftName.trimmingCharacters(in: .whitespacesAndNewlines),
            tags: parseTags(personDraftTags),
            notes: personDraftNotes.trimmingCharacters(in: .whitespacesAndNewlines),
            appearances: people[index].appearances
        )
        do {
            try store.savePersonTags(libraryURL: libraryURL, person: updated)
            people[index] = updated
            statusMessage = "Saved tags for \(updated.title)"
        } catch {
            statusMessage = "Could not save person tags: \(error.localizedDescription)"
        }
    }

    func matchingClipMoments(for segment: TranscriptSegment) -> [ClipMoment] {
        clipMoments.filter { clipMoment in
            clipMoment.relativePath == segment.relativePath && overlaps(clipMoment, segment)
        }
        .sorted { hookRank($0.hookStrength) > hookRank($1.hookStrength) }
    }

    func clipTags(for relativePath: String) -> ClipTag? {
        clipTags.first { $0.relativePath == relativePath }
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

    private func bestSegment(relativePath: String, timestamp: Double) -> TranscriptSegment? {
        let candidates = segments.filter { $0.relativePath == relativePath }
        if let containing = candidates.first(where: { $0.start <= timestamp && timestamp <= $0.end }) {
            return containing
        }
        return candidates.min { lhs, rhs in
            let lhsDistance = min(abs(lhs.start - timestamp), abs(lhs.end - timestamp))
            let rhsDistance = min(abs(rhs.start - timestamp), abs(rhs.end - timestamp))
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
        clipTags = []
        analysisArtifacts = []
        people = []
        selectedFileID = nil
        selectedSegmentID = nil
        selectedClipMomentID = nil
        selectedAnalysisArtifactID = nil
        selectedPersonID = nil
        selectedTagFilter = nil
        tagSearchText = ""
        libraryMode = .review
        resetPersonDrafts()
        player.pause()
        if let boundaryObserver {
            player.removeTimeObserver(boundaryObserver)
            self.boundaryObserver = nil
        }
        player.replaceCurrentItem(with: nil)
        currentPlayerURL = nil
    }

    private func loadPersonDrafts() {
        guard let selectedPerson else {
            resetPersonDrafts()
            return
        }
        personDraftName = selectedPerson.displayName
        personDraftTags = selectedPerson.tags.joined(separator: ", ")
        personDraftNotes = selectedPerson.notes
    }

    private func resetPersonDrafts() {
        personDraftName = ""
        personDraftTags = ""
        personDraftNotes = ""
    }

    private func parseTags(_ text: String) -> [String] {
        text.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func clipTagsMatch(relativePath: String, query: String) -> Bool {
        guard let clipTag = clipTags(for: relativePath) else { return false }
        return clipTag.tags.contains { $0.localizedCaseInsensitiveContains(query) }
            || clipTag.locationTags.contains { $0.localizedCaseInsensitiveContains(query) }
            || clipTag.spokenLanguageTags.contains { $0.localizedCaseInsensitiveContains(query) }
            || clipTag.themeTags.contains { $0.localizedCaseInsensitiveContains(query) }
            || clipTag.entityTags.contains { $0.localizedCaseInsensitiveContains(query) }
            || clipTag.interviewLanguageTags.contains { $0.localizedCaseInsensitiveContains(query) }
            || clipTag.qualityTags.contains { $0.localizedCaseInsensitiveContains(query) }
    }

    private func matchesSelectedTag(relativePath: String) -> Bool {
        guard let selectedTagFilter else { return true }
        guard let clipTag = clipTags(for: relativePath) else { return false }
        let selectedKey = tagKey(selectedTagFilter)
        return clipTag.displayTags.contains { tagKey($0) == selectedKey }
    }

    private func tagKey(_ tag: String) -> String {
        tag.trimmingCharacters(in: .whitespacesAndNewlines)
            .folding(options: [.caseInsensitive, .diacriticInsensitive], locale: .current)
            .lowercased()
    }
}
