import AVKit
import SwiftUI
import TranscriptViewerCore

struct RootView: View {
    let model: LibraryViewModel

    var body: some View {
        ZStack {
            NavigationSplitView {
                SidebarView(model: model)
                    .navigationSplitViewColumnWidth(min: 280, ideal: 330, max: 420)
            } content: {
                ReviewConsoleView(model: model)
                    .navigationSplitViewColumnWidth(min: 650, ideal: 820)
            } detail: {
                InspectorView(model: model)
                    .navigationSplitViewColumnWidth(min: 340, ideal: 390, max: 470)
            }
            .disabled(model.libraryURL == nil)

            if model.libraryURL == nil {
                WelcomeOverlay(model: model)
            }

            ShortcutLayer(model: model)
        }
        .toolbar {
            ToolbarItemGroup {
                Button {
                    NSApp.sendAction(#selector(TranscriptViewerApplication.openLibrary), to: nil, from: nil)
                } label: {
                    Label("Open Library", systemImage: "folder")
                }
                Button {
                    model.reload()
                } label: {
                    Label("Reload", systemImage: "arrow.clockwise")
                }
                .disabled(model.libraryURL == nil)
                Button {
                    model.inspectorMode = .aiPlan
                } label: {
                    Label("AI Plan", systemImage: "sparkles.rectangle.stack")
                }
                .disabled(model.analysisArtifacts.isEmpty && model.clipMoments.isEmpty)
            }
        }
        .safeAreaInset(edge: .bottom) {
            StatusBar(model: model)
        }
    }
}

struct WelcomeOverlay: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "waveform.and.magnifyingglass")
                .font(.system(size: 56, weight: .semibold))
                .foregroundStyle(.tint)
            VStack(spacing: 6) {
                Text("Transcript Viewer")
                    .font(.system(size: 32, weight: .bold))
                Text("Open a WhisperX _ai_library folder to browse AI-ranked video moments.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                Button {
                    NSApp.sendAction(#selector(TranscriptViewerApplication.openLibrary), to: nil, from: nil)
                } label: {
                    Label("Open Library", systemImage: "folder.badge.plus")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                if let path = model.recentLibraries.first {
                    Button {
                        Task { await model.loadLibrary(URL(fileURLWithPath: path)) }
                    } label: {
                        Label("Reopen Last", systemImage: "clock.arrow.circlepath")
                    }
                    .controlSize(.large)
                }
            }
            if !model.recentLibraries.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Recent Libraries")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    ForEach(model.recentLibraries, id: \.self) { path in
                        Button {
                            Task { await model.loadLibrary(URL(fileURLWithPath: path)) }
                        } label: {
                            HStack {
                                Image(systemName: "folder")
                                    .foregroundStyle(.secondary)
                                Text(path)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                                Spacer()
                            }
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 7))
                    }
                }
                .frame(width: 520)
                .padding(.top, 6)
            }
        }
        .padding(34)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.regularMaterial)
    }
}

struct SidebarView: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(spacing: 0) {
            LibraryHeader(model: model)
            Divider()
            FileSearchField(model: model)
            Divider()
            List {
                Section {
                    QueueRow(
                        title: "All transcripts",
                        subtitle: "\(model.segments.count) moments",
                        systemImage: "rectangle.stack",
                        isSelected: model.selectedFileID == nil
                    ) {
                        model.clearFileSelection()
                    }
                }
                Section("Files") {
                    ForEach(model.filteredFiles) { file in
                        FileRow(file: file, isSelected: model.selectedFileID == file.id) {
                            model.choose(file: file)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
        }
    }
}

struct LibraryHeader: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 11) {
                Image(systemName: "waveform.and.magnifyingglass")
                    .font(.title2)
                    .foregroundStyle(.tint)
                    .frame(width: 30)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Transcript Viewer")
                        .font(.headline)
                    Text(model.libraryURL?.path ?? "No library open")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
            }

            ProgressView(value: model.progressFraction)
                .progressViewStyle(.linear)

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                MetricTile(value: "\(model.doneFileCount)", label: "done")
                MetricTile(value: "\(model.segments.count)", label: "moments")
                MetricTile(value: "\(model.clipMoments.count)", label: "AI picks")
                MetricTile(value: "\(model.highPriorityClipMomentCount)", label: "high hook")
            }
        }
        .padding(14)
    }
}

struct MetricTile: View {
    var value: String
    var label: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(.title3, design: .rounded, weight: .semibold))
                .monospacedDigit()
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 7))
    }
}

struct FileSearchField: View {
    let model: LibraryViewModel

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Filter files", text: Binding(get: { model.fileSearchText }, set: { model.fileSearchText = $0 }))
                .textFieldStyle(.plain)
            if !model.fileSearchText.isEmpty {
                Button {
                    model.fileSearchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
    }
}

struct QueueRow: View {
    var title: String
    var subtitle: String
    var systemImage: String
    var isSelected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: systemImage)
                    .foregroundStyle(.tint)
                    .frame(width: 20)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 7)
            .background(isSelected ? Color.accentColor.opacity(0.15) : Color.clear, in: RoundedRectangle(cornerRadius: 7))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct FileRow: View {
    var file: TranscriptFile
    var isSelected: Bool
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: file.segmentCount == 0 ? "film.stack" : "film")
                    .foregroundStyle(file.status == "done" ? Color.accentColor : Color.secondary)
                    .frame(width: 20)
                VStack(alignment: .leading, spacing: 3) {
                    Text(file.relativePath)
                        .font(.callout.weight(.medium))
                        .lineLimit(1)
                    HStack(spacing: 5) {
                        StatusDot(status: file.status)
                        Text("\(file.segmentCount) moments")
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, 6)
            .padding(.horizontal, 7)
            .background(isSelected ? Color.accentColor.opacity(0.15) : Color.clear, in: RoundedRectangle(cornerRadius: 7))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

struct StatusDot: View {
    var status: String

    var body: some View {
        Circle()
            .fill(status == "done" ? Color.green : Color.secondary)
            .frame(width: 6, height: 6)
            .accessibilityLabel(status)
    }
}

struct ReviewConsoleView: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(spacing: 0) {
            ReviewHeader(model: model)
            Divider()
            PlayerPane(model: model)
            Divider()
            SegmentFilterBar(model: model)
            Divider()
            SegmentList(model: model)
        }
        .navigationTitle("Review")
    }
}

struct ReviewHeader: View {
    let model: LibraryViewModel

    var body: some View {
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.selectedFile?.relativePath ?? "All transcripts")
                    .font(.title3.weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(model.reviewPositionText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button {
                model.startAIAssistedReview()
            } label: {
                Label("AI Review", systemImage: "sparkles")
            }
            .buttonStyle(.bordered)
            .disabled(model.clipMoments.isEmpty)

            Button {
                model.focusHighHookAIPick()
            } label: {
                Label("High Hooks", systemImage: "flame.fill")
            }
            .buttonStyle(.bordered)
            .tint(.red)
            .disabled(model.highPriorityClipMomentCount == 0)

            Button {
                model.focusPreviousAIPick()
            } label: {
                Label("Previous AI Pick", systemImage: "chevron.up")
            }
            .labelStyle(.iconOnly)
            .help("Previous AI pick")
            .disabled(model.filteredClipMoments.isEmpty)

            Button {
                model.focusNextAIPick()
            } label: {
                Label("Next AI Pick", systemImage: "chevron.down")
            }
            .labelStyle(.iconOnly)
            .help("Next AI pick")
            .disabled(model.filteredClipMoments.isEmpty)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
    }
}

struct PlayerPane: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(spacing: 0) {
            ZStack(alignment: .topLeading) {
                NativePlayerView(player: model.player)
                    .background(Color.black)
                if let segment = model.selectedSegment {
                    PlayerOverlay(segment: segment, duration: model.selectedDurationText)
                }
            }
            .frame(minHeight: 300, idealHeight: 390, maxHeight: 500)

            HStack(spacing: 10) {
                Button {
                    model.focusPrevious()
                } label: {
                    Label("Previous Moment", systemImage: "chevron.up")
                }
                .help("Previous moment")
                .disabled(model.selectedSegmentIndex == 0 || model.selectedSegment == nil)

                Button {
                    model.nudgePlayback(seconds: -2)
                } label: {
                    Label("Back 2 Seconds", systemImage: "gobackward.2")
                }
                .help("Back 2 seconds")

                Button {
                    model.togglePlayback()
                } label: {
                    Label("Play/Pause", systemImage: "playpause.fill")
                }
                .help("Play or pause")

                Button {
                    model.nudgePlayback(seconds: 2)
                } label: {
                    Label("Forward 2 Seconds", systemImage: "goforward.2")
                }
                .help("Forward 2 seconds")

                Button {
                    model.focusNext()
                } label: {
                    Label("Next Moment", systemImage: "chevron.down")
                }
                .help("Next moment")
                .disabled(model.selectedSegmentIndex == model.filteredSegments.count - 1 || model.selectedSegment == nil)

                Divider()
                    .frame(height: 22)

                Text(model.selectedDurationText)
                    .font(.callout)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)

                Spacer()

                if let pick = model.currentAIPick {
                    HookBadge(rank: model.hookRank(pick.hookStrength), label: pick.hookStrength)
                }
            }
            .labelStyle(.iconOnly)
            .padding(.horizontal, 14)
            .padding(.vertical, 9)
            .background(.bar)
        }
    }
}

struct PlayerOverlay: View {
    var segment: TranscriptSegment
    var duration: String

    var body: some View {
        HStack(spacing: 10) {
            Text(segment.relativePath)
                .lineLimit(1)
                .truncationMode(.middle)
            Text(duration)
                .monospacedDigit()
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(.black.opacity(0.58), in: Capsule())
        .padding(12)
    }
}

struct NativePlayerView: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .floating
        view.videoGravity = .resizeAspect
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}

struct SegmentFilterBar: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(spacing: 9) {
            HStack(spacing: 10) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search transcript, speaker, file", text: Binding(
                    get: { model.searchText },
                    set: { model.searchText = $0 }
                ))
                .textFieldStyle(.plain)
                if !model.searchText.isEmpty {
                    Button {
                        model.searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
            }
            Picker("Scope", selection: Binding(
                get: { model.segmentScope },
                set: { model.setScope($0) }
            )) {
                ForEach(LibraryViewModel.SegmentScope.allCases) { scope in
                    Label("\(scope.title) \(model.count(for: scope))", systemImage: scope.systemImage).tag(scope)
                }
            }
            .pickerStyle(.segmented)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
    }
}

struct SegmentList: View {
    let model: LibraryViewModel

    var body: some View {
        if model.filteredSegments.isEmpty {
            ContentUnavailableView("No transcript moments", systemImage: "text.magnifyingglass")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            List(selection: segmentSelection) {
                ForEach(model.filteredSegments) { segment in
                    SegmentRow(
                        segment: segment,
                        isSelected: model.selectedSegmentID == segment.id,
                        formatTime: model.formatTime
                    )
                    .tag(segment.id)
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 4, leading: 10, bottom: 4, trailing: 10))
                    .contentShape(Rectangle())
                    .onTapGesture {
                        model.focus(segment, autoplay: true)
                    }
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
        }
    }

    private var segmentSelection: Binding<String?> {
        Binding(
            get: { model.selectedSegmentID },
            set: { newValue in
                guard let newValue, let segment = model.segments.first(where: { $0.id == newValue }) else { return }
                model.focus(segment, autoplay: true)
            }
        )
    }
}

struct SegmentRow: View {
    var segment: TranscriptSegment
    var isSelected: Bool
    var formatTime: (Double) -> String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            RoundedRectangle(cornerRadius: 2)
                .fill(isSelected ? Color.accentColor : Color.clear)
                .frame(width: 4)
                .padding(.vertical, 4)
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 8) {
                    Text("\(formatTime(segment.start)) - \(formatTime(segment.end))")
                        .monospacedDigit()
                    Text(segment.speaker ?? "Unknown speaker")
                    Text(segment.relativePath)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                }
                .font(.caption)
                .foregroundStyle(.secondary)

                Text(segment.text.isEmpty ? "No transcript text" : segment.text)
                    .font(.body)
                    .lineLimit(5)
                    .textSelection(.enabled)
                    .foregroundStyle(segment.text.isEmpty ? .secondary : .primary)
            }
        }
        .padding(10)
        .background(isSelected ? Color.accentColor.opacity(0.10) : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(isSelected ? Color.accentColor.opacity(0.45) : Color.secondary.opacity(0.10), lineWidth: 1)
        }
    }
}

struct InspectorView: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(spacing: 0) {
            Picker("Inspector", selection: Binding(get: { model.inspectorMode }, set: { model.inspectorMode = $0 })) {
                ForEach(LibraryViewModel.InspectorMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)
            .padding(12)

            Divider()

            switch model.inspectorMode {
            case .moment:
                MomentInspectorView(model: model)
            case .aiPlan:
                AIPlanInspectorView(model: model)
            }
        }
        .navigationTitle("Inspector")
    }
}

struct MomentInspectorView: View {
    let model: LibraryViewModel

    var body: some View {
        if let segment = model.selectedSegment {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    InspectorSummary(segment: segment, model: model)
                    AIMatchesPanel(model: model, segment: segment)
                    AIRecommendationPanel(model: model)
                    TranscriptPanel(segment: segment)
                }
                .padding(16)
            }
            Divider()
            InspectorFooter(model: model)
        } else {
            ContentUnavailableView("Select a transcript moment", systemImage: "sidebar.right")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }
}

struct AIRecommendationPanel: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("AI Recommendation", systemImage: "sparkles")
                .font(.headline)
            if let pick = model.currentAIPick {
                HStack {
                    HookBadge(rank: model.hookRank(pick.hookStrength), label: pick.hookStrength)
                    Text("\(model.formatTime(pick.start)) - \(model.formatTime(pick.end))")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                    Spacer()
                }
                if !pick.theme.isEmpty {
                    Text(pick.theme)
                        .font(.callout.weight(.semibold))
                }
                Text(pick.text)
                    .font(.callout)
                    .lineLimit(6)
                    .textSelection(.enabled)
            } else {
                ContentUnavailableView("No matched AI pick", systemImage: "sparkle.magnifyingglass")
                    .frame(maxWidth: .infinity, minHeight: 120)
            }
        }
        .panelStyle()
    }
}

struct AIPlanInspectorView: View {
    let model: LibraryViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                AIPlanSummary(model: model)
                AIPicksPanel(model: model)
                AnalysisArtifactsPanel(model: model)
            }
            .padding(16)
        }
    }
}

struct AIPlanSummary: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("AI Plan", systemImage: "sparkles.rectangle.stack")
                .font(.headline)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                MetricTile(value: "\(model.clipMoments.count)", label: "ranked picks")
                MetricTile(value: "\(model.highPriorityClipMomentCount)", label: "high hook")
                MetricTile(value: "\(model.analysisArtifacts.count)", label: "notes")
                MetricTile(value: "\(model.clipThemes.count - 1)", label: "themes")
            }
        }
        .panelStyle()
    }
}

struct AIPicksPanel: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label(model.clipFilterSummary, systemImage: "quote.bubble")
                    .font(.headline)
                Spacer()
            }
            VStack(spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search AI picks", text: Binding(get: { model.clipSearchText }, set: { model.clipSearchText = $0 }))
                        .textFieldStyle(.plain)
                    if !model.clipSearchText.isEmpty {
                        Button {
                            model.clipSearchText = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 7))

                Picker("Theme", selection: Binding(get: { model.clipThemeFilter }, set: { model.clipThemeFilter = $0 })) {
                    ForEach(model.clipThemes, id: \.self) { theme in
                        Text(theme).tag(theme)
                    }
                }
            }

            if model.filteredClipMoments.isEmpty {
                ContentUnavailableView("No AI picks", systemImage: "sparkle.magnifyingglass")
                    .frame(maxWidth: .infinity, minHeight: 120)
            } else {
                LazyVStack(spacing: 10) {
                    ForEach(model.filteredClipMoments.prefix(80)) { clipMoment in
                        ClipMomentCard(
                            clipMoment: clipMoment,
                            isSelected: model.selectedClipMomentID == clipMoment.id,
                            hookRank: model.hookRank(clipMoment.hookStrength),
                            formatTime: model.formatTime,
                            play: { model.focus(clipMoment, autoplay: true) }
                        )
                    }
                }
            }
        }
        .panelStyle()
    }
}

struct ClipMomentCard: View {
    var clipMoment: ClipMoment
    var isSelected: Bool
    var hookRank: Int
    var formatTime: (Double) -> String
    var play: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack(spacing: 8) {
                HookBadge(rank: hookRank, label: clipMoment.hookStrength)
                Text("\(formatTime(clipMoment.start)) - \(formatTime(clipMoment.end))")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
                Button(action: play) {
                    Label("Play", systemImage: "play.fill")
                }
                .labelStyle(.iconOnly)
                .help("Play AI pick")
            }
            Text(clipMoment.relativePath)
                .font(.caption.weight(.semibold))
                .lineLimit(1)
                .truncationMode(.middle)
            if !clipMoment.theme.isEmpty {
                Text(clipMoment.theme)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            Text(clipMoment.text.isEmpty ? "No transcript text" : clipMoment.text)
                .font(.callout)
                .lineLimit(5)
                .textSelection(.enabled)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(isSelected ? Color.accentColor.opacity(0.12) : Color(nsColor: .windowBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .strokeBorder(isSelected ? Color.accentColor.opacity(0.5) : Color.secondary.opacity(0.12), lineWidth: 1)
        }
    }
}

struct HookBadge: View {
    var rank: Int
    var label: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "flame.fill")
            Text(displayLabel)
        }
        .font(.caption2.weight(.semibold))
        .foregroundStyle(color)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(color.opacity(0.13), in: Capsule())
    }

    private var displayLabel: String {
        label.isEmpty ? "Hook \(rank)" : label.capitalized
    }

    private var color: Color {
        switch rank {
        case 5...:
            .red
        case 4:
            .orange
        default:
            .accentColor
        }
    }
}

struct AnalysisArtifactsPanel: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Generated Notes", systemImage: "doc.text.magnifyingglass")
                .font(.headline)
            if model.analysisArtifacts.isEmpty {
                ContentUnavailableView("No analysis notes", systemImage: "doc.text")
                    .frame(maxWidth: .infinity, minHeight: 120)
            } else {
                Picker("Note", selection: Binding(get: {
                    model.selectedAnalysisArtifactID ?? model.analysisArtifacts.first?.id ?? ""
                }, set: { model.selectedAnalysisArtifactID = $0 })) {
                    ForEach(model.analysisArtifacts) { artifact in
                        Text(artifact.title).tag(artifact.id)
                    }
                }
                if let artifact = model.selectedAnalysisArtifact {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(artifact.filename)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ScrollView {
                            Text(artifact.content)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(minHeight: 220, maxHeight: 360)
                        .padding(10)
                        .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 7))
                        .overlay {
                            RoundedRectangle(cornerRadius: 7)
                                .strokeBorder(Color.secondary.opacity(0.14), lineWidth: 1)
                        }
                    }
                }
            }
        }
        .panelStyle()
    }
}

struct AIMatchesPanel: View {
    let model: LibraryViewModel
    var segment: TranscriptSegment

    var body: some View {
        let matches = model.matchingClipMoments(for: segment)
        if !matches.isEmpty {
            VStack(alignment: .leading, spacing: 10) {
                Label("AI Matches", systemImage: "sparkles")
                    .font(.headline)
                ForEach(matches.prefix(3)) { clipMoment in
                    HStack(alignment: .top, spacing: 9) {
                        HookBadge(rank: model.hookRank(clipMoment.hookStrength), label: clipMoment.hookStrength)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(clipMoment.theme.isEmpty ? "Untitled theme" : clipMoment.theme)
                                .font(.caption.weight(.semibold))
                                .lineLimit(1)
                            Text(clipMoment.text)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(3)
                        }
                    }
                }
            }
            .panelStyle()
        }
    }
}

struct InspectorSummary: View {
    var segment: TranscriptSegment
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Selected Moment")
                        .font(.headline)
                    Text(segment.relativePath)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                Spacer()
            }
            Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                GridRow {
                    DetailLabel("Speaker")
                    Text(segment.speaker ?? "Unknown")
                }
                GridRow {
                    DetailLabel("Start")
                    Text(model.formatTime(segment.start)).monospacedDigit()
                }
                GridRow {
                    DetailLabel("End")
                    Text(model.formatTime(segment.end)).monospacedDigit()
                }
            }
            .font(.callout)
        }
        .panelStyle()
    }
}

struct DetailLabel: View {
    var title: String

    init(_ title: String) {
        self.title = title
    }

    var body: some View {
        Text(title)
            .foregroundStyle(.secondary)
            .frame(width: 70, alignment: .leading)
    }
}

struct TranscriptPanel: View {
    var segment: TranscriptSegment

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Transcript")
                .font(.headline)
            Text(segment.text.isEmpty ? "No transcript text." : segment.text)
                .font(.body)
                .lineSpacing(4)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .panelStyle()
    }
}

struct InspectorFooter: View {
    let model: LibraryViewModel

    var body: some View {
        HStack(spacing: 10) {
            Button {
                model.revealSourceInFinder()
            } label: {
                Label("Reveal Source", systemImage: "folder")
            }
            .labelStyle(.iconOnly)
            .help("Reveal source video in Finder")
            .disabled(model.selectedSourcePath.isEmpty)

            Spacer()
        }
        .padding(12)
        .padding(.bottom, 24)
    }
}

struct StatusBar: View {
    let model: LibraryViewModel

    var body: some View {
        HStack(spacing: 10) {
            if model.isLoading {
                ProgressView()
                    .controlSize(.small)
            }
            Text(model.statusMessage)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer()
            Text("\(model.files.count) files")
            Text("\(model.segments.count) moments")
            Text("\(model.clipMoments.count) AI picks")
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.bar)
    }
}

struct ShortcutLayer: View {
    let model: LibraryViewModel

    var body: some View {
        VStack {
            Button("Play/Pause") { model.togglePlayback() }
                .keyboardShortcut(.space, modifiers: [])
            Button("Start AI Review") { model.startAIAssistedReview() }
                .keyboardShortcut("b", modifiers: [])
            Button("AI Plan") { model.inspectorMode = .aiPlan }
                .keyboardShortcut("a", modifiers: [])
            Button("Previous AI Pick") { model.focusPreviousAIPick() }
                .keyboardShortcut(.upArrow, modifiers: [])
            Button("Next AI Pick") { model.focusNextAIPick() }
                .keyboardShortcut(.downArrow, modifiers: [])
        }
        .frame(width: 0, height: 0)
        .opacity(0.01)
        .accessibilityHidden(true)
    }
}

private struct PanelStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(Color.secondary.opacity(0.12))
            }
    }
}

private extension View {
    func panelStyle() -> some View {
        modifier(PanelStyle())
    }
}
