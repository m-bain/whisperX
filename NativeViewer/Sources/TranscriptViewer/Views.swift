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
                    model.exportCSV()
                } label: {
                    Label("Export CSV", systemImage: "square.and.arrow.down")
                }
                .disabled(model.selects.isEmpty)
                Button {
                    model.revealExportInFinder()
                } label: {
                    Label("Reveal Export", systemImage: "folder.badge.gearshape")
                }
                .disabled(model.selects.isEmpty)
                Button {
                    model.showSelectsShelf.toggle()
                } label: {
                    Label("Selects Shelf", systemImage: "rectangle.bottomthird.inset.filled")
                }
                .disabled(model.libraryURL == nil)
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
                Text("Open a WhisperX _ai_library folder to review, rank, and export video moments.")
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
                        FileRow(
                            file: file,
                            selectCount: model.selectCount(for: file),
                            isSelected: model.selectedFileID == file.id
                        ) {
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
                    Text("Transcript Selects")
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
                MetricTile(value: "\(model.reviewedCount)", label: "reviewed")
                MetricTile(value: "\(model.unreviewedCount)", label: "left")
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
    var selectCount: Int
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
                        if selectCount > 0 {
                            Text("· \(selectCount) selects")
                        }
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
            if model.showSelectsShelf, !model.selects.isEmpty {
                Divider()
                SelectsShelfView(model: model)
            }
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
                model.startUnreviewedReview()
            } label: {
                Label("Review", systemImage: "play.circle")
            }
            .buttonStyle(.bordered)
            .disabled(model.unreviewedCount == 0)
            QuickMarkButton(title: "Good", systemImage: "checkmark.circle.fill", color: .green) {
                model.mark(status: .good, hookStrength: 5, advance: model.autoAdvanceAfterMark)
            }
            .disabled(model.selectedSegment == nil)
            QuickMarkButton(title: "Maybe", systemImage: "questionmark.circle.fill", color: .orange) {
                model.mark(status: .maybe, advance: model.autoAdvanceAfterMark)
            }
            .disabled(model.selectedSegment == nil)
            QuickMarkButton(title: "Weak", systemImage: "xmark.circle.fill", color: .red) {
                model.mark(status: .weak, advance: model.autoAdvanceAfterMark)
            }
            .disabled(model.selectedSegment == nil)
            Toggle(isOn: Binding(get: { model.autoAdvanceAfterMark }, set: { model.autoAdvanceAfterMark = $0 })) {
                Label("Auto Advance", systemImage: "arrow.down.circle")
            }
            .toggleStyle(.switch)
            .controlSize(.small)
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 12)
    }
}

struct QuickMarkButton: View {
    var title: String
    var systemImage: String
    var color: Color
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .labelStyle(.titleAndIcon)
        }
        .buttonStyle(.bordered)
        .tint(color)
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
                    PlayerOverlay(segment: segment, select: model.select(for: segment), duration: model.selectedDurationText)
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

                HookStrengthPicker(model: model)
                    .disabled(model.selectedSegment == nil)
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
    var select: SelectMoment?
    var duration: String

    var body: some View {
        HStack(spacing: 10) {
            Text(segment.relativePath)
                .lineLimit(1)
                .truncationMode(.middle)
            Text(duration)
                .monospacedDigit()
            if let select {
                SelectBadge(select: select)
            }
        }
        .font(.caption.weight(.semibold))
        .foregroundStyle(.white)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(.black.opacity(0.58), in: Capsule())
        .padding(12)
    }
}

struct HookStrengthPicker: View {
    let model: LibraryViewModel

    var body: some View {
        HStack(spacing: 4) {
            Text("Hook")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(1...5, id: \.self) { value in
                Button {
                    model.setHookStrength(value)
                } label: {
                    Image(systemName: value <= model.draftHookStrength ? "star.fill" : "star")
                }
                .buttonStyle(.plain)
                .foregroundStyle(value <= model.draftHookStrength ? Color.yellow : Color.secondary)
                .help("Hook strength \(value)")
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 5)
        .background(.quaternary, in: Capsule())
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

struct SelectsShelfView: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("\(model.selects.count) Selects", systemImage: "tray.full")
                    .font(.headline)
                Spacer()
                Button {
                    model.exportCSV()
                } label: {
                    Label("Export", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)
                Button {
                    model.showSelectsShelf = false
                } label: {
                    Label("Hide", systemImage: "chevron.down")
                }
                .labelStyle(.iconOnly)
                .help("Hide selects shelf")
            }

            ScrollView(.horizontal, showsIndicators: false) {
                LazyHStack(spacing: 10) {
                    ForEach(model.sortedSelects) { select in
                        SavedSelectCard(
                            select: select,
                            isSelected: model.selectedSegmentID == select.segmentID,
                            formatTime: model.formatTime
                        ) {
                            model.focus(select, autoplay: true)
                        }
                    }
                }
                .padding(.bottom, 2)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .frame(height: 154)
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

struct SavedSelectCard: View {
    var select: SelectMoment
    var isSelected: Bool
    var formatTime: (Double) -> String
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 6) {
                    SelectBadge(select: select)
                    Spacer(minLength: 0)
                    Text("\(formatTime(select.start)) - \(formatTime(select.end))")
                        .font(.caption2)
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Text(select.relativePath)
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(select.text)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(10)
            .frame(width: 260, height: 102, alignment: .topLeading)
            .background(isSelected ? Color.accentColor.opacity(0.13) : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .strokeBorder(isSelected ? Color.accentColor.opacity(0.55) : Color.secondary.opacity(0.12), lineWidth: 1)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
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
                        select: model.select(for: segment),
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
    var select: SelectMoment?
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
                    if let select {
                        SelectBadge(select: select)
                    }
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

struct SelectBadge: View {
    var select: SelectMoment

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
            Text(select.hookStrength.map { "Hook \($0)" } ?? select.status.title)
        }
        .font(.caption2.weight(.semibold))
        .foregroundStyle(foreground)
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(foreground.opacity(0.13), in: Capsule())
    }

    private var icon: String {
        switch select.status {
        case .good: "checkmark.circle.fill"
        case .maybe: "questionmark.circle.fill"
        case .weak, .unusable: "xmark.circle.fill"
        case .selected: "pin.fill"
        }
    }

    private var foreground: Color {
        switch select.status {
        case .good: .green
        case .maybe: .orange
        case .weak, .unusable: .red
        case .selected: .accentColor
        }
    }
}

struct InspectorView: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(spacing: 0) {
            if let segment = model.selectedSegment {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        InspectorSummary(segment: segment, select: model.select(for: segment), model: model)
                        EditorialPanel(model: model)
                        TranscriptPanel(segment: segment)
                    }
                    .padding(16)
                }
                Divider()
                InspectorFooter(model: model, segment: segment)
            } else {
                ContentUnavailableView("Select a transcript moment", systemImage: "sidebar.right")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("Inspector")
    }
}

struct InspectorSummary: View {
    var segment: TranscriptSegment
    var select: SelectMoment?
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
                if let select {
                    SelectBadge(select: select)
                }
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

struct EditorialPanel: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Editorial")
                .font(.headline)

            StatusSelector(model: model)

            VStack(alignment: .leading, spacing: 7) {
                Text("Hook Strength")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                HookStrengthPicker(model: model)
            }

            LabeledContent("Start") {
                TrimField(
                    value: Binding(get: { model.draftStart }, set: { model.draftStart = $0 }),
                    decrement: { model.adjustDraftStart(by: -0.1) },
                    increment: { model.adjustDraftStart(by: 0.1) }
                )
            }
            LabeledContent("End") {
                TrimField(
                    value: Binding(get: { model.draftEnd }, set: { model.draftEnd = $0 }),
                    decrement: { model.adjustDraftEnd(by: -0.1) },
                    increment: { model.adjustDraftEnd(by: 0.1) }
                )
            }
            TextField("Theme", text: Binding(get: { model.draftTheme }, set: { model.draftTheme = $0 }))
                .textFieldStyle(.roundedBorder)
            TextField("Tags", text: Binding(get: { model.draftTags }, set: { model.draftTags = $0 }))
                .textFieldStyle(.roundedBorder)
            VStack(alignment: .leading, spacing: 6) {
                Text("Notes")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                TextEditor(text: Binding(get: { model.draftNotes }, set: { model.draftNotes = $0 }))
                    .font(.body)
                    .frame(minHeight: 95)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 7))
                    .overlay {
                        RoundedRectangle(cornerRadius: 7)
                            .strokeBorder(Color.secondary.opacity(0.18))
                    }
            }
        }
        .panelStyle()
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
    var segment: TranscriptSegment

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

            Button(role: .destructive) {
                model.deleteSelectedSelect()
            } label: {
                Label("Remove", systemImage: "trash")
            }
            .labelStyle(.iconOnly)
            .help("Remove saved select")
            .disabled(model.select(for: segment) == nil)

            Spacer()

            Button {
                model.copySelectedSelectCSV()
            } label: {
                Label("Copy CSV", systemImage: "doc.on.doc")
            }
            .labelStyle(.iconOnly)
            .help("Copy saved select as CSV row")
            .disabled(!model.hasSelectedSelect)

            Button {
                model.saveDraft()
            } label: {
                Label("Save", systemImage: "tray.and.arrow.down.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.selectedSegment == nil)
        }
        .padding(12)
        .padding(.bottom, 24)
    }
}

struct TrimField: View {
    @Binding var value: String
    var decrement: () -> Void
    var increment: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            Button(action: decrement) {
                Image(systemName: "minus")
            }
            .buttonStyle(.borderless)
            .help("Nudge earlier")

            TextField("Time", text: $value)
                .textFieldStyle(.roundedBorder)
                .monospacedDigit()

            Button(action: increment) {
                Image(systemName: "plus")
            }
            .buttonStyle(.borderless)
            .help("Nudge later")
        }
    }
}

struct StatusSelector: View {
    let model: LibraryViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Status")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                ForEach(SelectStatus.allCases, id: \.self) { status in
                    Button {
                        model.draftStatus = status
                    } label: {
                        Text(status.title)
                            .font(.caption.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 7)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(model.draftStatus == status ? .white : .primary)
                    .background(statusColor(status, selected: model.draftStatus == status), in: RoundedRectangle(cornerRadius: 7))
                    .overlay {
                        RoundedRectangle(cornerRadius: 7)
                            .strokeBorder(Color.secondary.opacity(model.draftStatus == status ? 0 : 0.18))
                    }
                }
            }
        }
    }

    private func statusColor(_ status: SelectStatus, selected: Bool) -> Color {
        guard selected else { return Color(nsColor: .controlBackgroundColor) }
        switch status {
        case .good: return .green
        case .maybe: return .orange
        case .weak, .unusable: return .red
        case .selected: return .accentColor
        }
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
            Text("\(model.selects.count) selects")
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
            Button("Start Review") { model.startUnreviewedReview() }
                .keyboardShortcut("b", modifiers: [])
            Button("Toggle Selects") { model.showSelectsShelf.toggle() }
                .keyboardShortcut("l", modifiers: [])
            Button("Previous") { model.focusPrevious() }
                .keyboardShortcut(.upArrow, modifiers: [])
            Button("Next") { model.focusNext() }
                .keyboardShortcut(.downArrow, modifiers: [])
            Button("Mark Good") { model.mark(status: .good, hookStrength: 5, advance: model.autoAdvanceAfterMark) }
                .keyboardShortcut("g", modifiers: [])
            Button("Mark Maybe") { model.mark(status: .maybe, advance: model.autoAdvanceAfterMark) }
                .keyboardShortcut("m", modifiers: [])
            Button("Mark Weak") { model.mark(status: .weak, advance: model.autoAdvanceAfterMark) }
                .keyboardShortcut("x", modifiers: [])
            Button("Mark Unusable") { model.mark(status: .unusable, hookStrength: 1, advance: model.autoAdvanceAfterMark) }
                .keyboardShortcut("u", modifiers: [])
            Button("Save") { model.saveDraft() }
                .keyboardShortcut("s", modifiers: [])
            Button("Copy Select CSV") { model.copySelectedSelectCSV() }
                .keyboardShortcut("c", modifiers: [])
            ForEach(1...5, id: \.self) { value in
                Button("Hook \(value)") { model.setHookStrength(value) }
                    .keyboardShortcut(KeyEquivalent(Character("\(value)")), modifiers: [])
            }
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
