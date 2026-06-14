import Foundation

public enum LibraryStoreError: LocalizedError {
    case missingManifest(URL)
    case invalidManifest([String])
    case missingColumn(String)
    case unreadableTranscript(URL)

    public var errorDescription: String? {
        switch self {
        case .missingManifest(let url):
            "No manifest.csv found at \(url.path)"
        case .invalidManifest(let missing):
            "manifest.csv is missing required columns: \(missing.joined(separator: ", "))"
        case .missingColumn(let name):
            "manifest.csv row is missing \(name)"
        case .unreadableTranscript(let url):
            "Could not read transcript JSON at \(url.path)"
        }
    }
}

public struct LibraryStore: Sendable {
    public static let clipMomentsFilename = "clip_moments.csv"
    public static let clipTagsFilename = "clip_tags.csv"
    public static let peopleIndexFilename = "people_index.csv"
    public static let peopleTagsFilename = "people_tags.csv"

    private static let analysisFiles: [(filename: String, fallbackTitle: String)] = [
        ("master_theme_map.md", "Master Theme Map"),
        ("best_50_quote_moments.md", "Best 50 Quote Moments"),
        ("top_20_short_form_clip_ideas.md", "Top 20 Short-Form Clip Ideas"),
        ("suggested_multi_person_sequences.md", "Suggested Multi-Person Sequences"),
        ("contradictory_answers_that_cut_well_together.md", "Contradictory Answers"),
        ("funny_surprising_emotional_moments.md", "Funny / Surprising / Emotional Moments"),
        ("weak_or_unusable_clips.md", "Weak Or Unusable Clips")
    ]

    public init() {}

    public func load(libraryURL: URL) throws -> LibrarySnapshot {
        let libraryURL = libraryURL.standardizedFileURL
        let filesAndSegments = try loadFilesAndSegments(libraryURL: libraryURL)
        let clipMoments = try loadClipMoments(libraryURL: libraryURL, files: filesAndSegments.files)
        let clipTags = try loadClipTags(libraryURL: libraryURL)
        let analysisArtifacts = try loadAnalysisArtifacts(libraryURL: libraryURL)
        let people = try loadPeople(libraryURL: libraryURL, files: filesAndSegments.files)
        return LibrarySnapshot(
            libraryURL: libraryURL,
            files: filesAndSegments.files,
            segments: filesAndSegments.segments,
            clipMoments: clipMoments,
            clipTags: clipTags,
            analysisArtifacts: analysisArtifacts,
            people: people
        )
    }

    public func replacePeopleIndex(libraryURL: URL, appearances: [PersonAppearance]) throws {
        let rows = [
            [
                "person_id",
                "appearance_id",
                "file",
                "file_id",
                "timestamp",
                "bbox_x",
                "bbox_y",
                "bbox_width",
                "bbox_height",
                "signature"
            ]
        ] + appearances.sorted {
            if $0.relativePath == $1.relativePath {
                return $0.timestamp < $1.timestamp
            }
            return $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending
        }.map { appearance in
            [
                appearance.personID,
                appearance.id,
                appearance.relativePath,
                appearance.fileID,
                Self.formatNumber(appearance.timestamp),
                Self.formatNumber(appearance.boundingBox.x),
                Self.formatNumber(appearance.boundingBox.y),
                Self.formatNumber(appearance.boundingBox.width),
                Self.formatNumber(appearance.boundingBox.height),
                appearance.signature
            ]
        }
        try CSV.encode(rows: rows)
            .write(
                to: libraryURL.standardizedFileURL.appendingPathComponent(Self.peopleIndexFilename),
                atomically: true,
                encoding: .utf8
            )
    }

    public func savePersonTags(libraryURL: URL, person: PersonProfile) throws {
        let url = libraryURL.standardizedFileURL.appendingPathComponent(Self.peopleTagsFilename)
        let existingRows = (try? String(contentsOf: url, encoding: .utf8))
            .map(CSV.parse)?
            .filter { !$0.allSatisfy(\.isEmpty) } ?? []
        let header = existingRows.first ?? ["person_id", "display_name", "tags", "notes", "updated_at"]
        let required = ["person_id", "display_name", "tags", "notes", "updated_at"]
        let outputHeader = required.allSatisfy { header.contains($0) } ? header : required
        let rows = existingRows.dropFirst().filter { row in
            guard let personIDIndex = outputHeader.firstIndex(of: "person_id"), personIDIndex < row.count else {
                return false
            }
            return row[personIDIndex] != person.id
        }
        let values = Dictionary(uniqueKeysWithValues: outputHeader.map { column in
            (column, tagValue(for: column, person: person))
        })
        let updatedRow = outputHeader.map { values[$0] ?? "" }
        try CSV.encode(rows: [outputHeader] + rows + [updatedRow])
            .write(to: url, atomically: true, encoding: .utf8)
    }

    private func loadFilesAndSegments(libraryURL: URL) throws -> (files: [TranscriptFile], segments: [TranscriptSegment]) {
        let manifestURL = libraryURL.appendingPathComponent("manifest.csv")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw LibraryStoreError.missingManifest(manifestURL)
        }

        let text = try String(contentsOf: manifestURL, encoding: .utf8)
        let rows = CSV.parse(text).filter { !$0.allSatisfy(\.isEmpty) }
        guard let header = rows.first else {
            throw LibraryStoreError.invalidManifest(["file", "relative_file", "output_dir", "json", "status"])
        }
        let required = ["file", "relative_file", "output_dir", "json", "status"]
        let missing = required.filter { !header.contains($0) }
        guard missing.isEmpty else {
            throw LibraryStoreError.invalidManifest(missing)
        }

        var files: [TranscriptFile] = []
        var allSegments: [TranscriptSegment] = []
        for row in rows.dropFirst() {
            let values = Dictionary(uniqueKeysWithValues: header.enumerated().map { index, column in
                (column, index < row.count ? row[index] : "")
            })
            let sourcePath = try requiredValue("file", in: values)
            let relativePath = try requiredValue("relative_file", in: values)
            let outputPath = try requiredValue("output_dir", in: values)
            let jsonPath = try requiredValue("json", in: values)
            let status = try requiredValue("status", in: values)
            let sourceURL = URL(fileURLWithPath: sourcePath).standardizedFileURL
            let jsonURL = URL(fileURLWithPath: jsonPath).standardizedFileURL
            let fileID = stableFileID(sourceURL.path)
            let transcript = status == "done" ? try? readTranscript(jsonURL: jsonURL) : nil
            let language = transcript?.language
            let segments = transcript?.segments.enumerated().map { index, segment in
                TranscriptSegment(
                    id: "\(fileID)::\(index)",
                    fileID: fileID,
                    sourceURL: sourceURL,
                    relativePath: relativePath,
                    index: index,
                    start: segment.start ?? 0,
                    end: segment.end ?? segment.start ?? 0,
                    speaker: segment.speaker,
                    text: segment.text.trimmingCharacters(in: .whitespacesAndNewlines),
                    language: language,
                    averageLogProbability: segment.averageLogProbability
                )
            } ?? []
            files.append(
                TranscriptFile(
                    id: fileID,
                    sourceURL: sourceURL,
                    relativePath: relativePath,
                    outputDirectory: URL(fileURLWithPath: outputPath).standardizedFileURL,
                    jsonURL: jsonURL,
                    status: status,
                    language: language,
                    segmentCount: segments.count
                )
            )
            allSegments.append(contentsOf: segments)
        }

        files.sort { $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending }
        allSegments.sort {
            if $0.relativePath == $1.relativePath {
                return $0.index < $1.index
            }
            return $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending
        }
        return (files, allSegments)
    }

    private func loadClipMoments(libraryURL: URL, files: [TranscriptFile]) throws -> [ClipMoment] {
        let url = libraryURL.appendingPathComponent(Self.clipMomentsFilename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }

        let text = try String(contentsOf: url, encoding: .utf8)
        let rows = CSV.parse(text).filter { !$0.allSatisfy(\.isEmpty) }
        guard let header = rows.first else { return [] }
        let sourceByRelativePath = Dictionary(uniqueKeysWithValues: files.map { ($0.relativePath, $0.sourceURL) })
        return rows.dropFirst().compactMap { row in
            let values = Dictionary(uniqueKeysWithValues: header.enumerated().map { index, column in
                (column, index < row.count ? row[index] : "")
            })
            guard
                let relativePath = nonEmpty(values["file"]),
                let startText = nonEmpty(values["start_time"]),
                let endText = nonEmpty(values["end_time"]),
                let start = parseTimecode(startText),
                let end = parseTimecode(endText)
            else {
                return nil
            }
            let theme = values["theme"] ?? ""
            let hookStrength = values["hook_strength"] ?? ""
            let quality = values["quality"] ?? ""
            let speaker = nonEmpty(values["speaker"])
            let body = values["text"] ?? ""
            return ClipMoment(
                id: stableID(relativePath, start, end, body),
                relativePath: relativePath,
                sourceURL: sourceByRelativePath[relativePath],
                start: start,
                end: max(start, end),
                theme: theme,
                hookStrength: hookStrength,
                quality: quality,
                speaker: speaker,
                text: body
            )
        }
    }

    private func loadClipTags(libraryURL: URL) throws -> [ClipTag] {
        let url = libraryURL.appendingPathComponent(Self.clipTagsFilename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }

        let text = try String(contentsOf: url, encoding: .utf8)
        let rows = CSV.parse(text).filter { !$0.allSatisfy(\.isEmpty) }
        guard let header = rows.first else { return [] }
        return rows.dropFirst().compactMap { row in
            let values = Dictionary(uniqueKeysWithValues: header.enumerated().map { index, column in
                (column, index < row.count ? row[index] : "")
            })
            guard let relativePath = nonEmpty(values["relative_file"] ?? values["file"]) else {
                return nil
            }
            let allTags = parseSemicolonTags(values["tags"])
            return ClipTag(
                id: stableID(relativePath),
                relativePath: relativePath,
                locationTags: parseSemicolonTags(values["location_tags"]),
                spokenLanguageTags: parseSemicolonTags(values["spoken_language_tags"]),
                themeTags: parseSemicolonTags(values["theme_tags"]),
                entityTags: parseSemicolonTags(values["entity_tags"]),
                interviewLanguageTags: parseSemicolonTags(values["interview_language_tags"]),
                qualityTags: parseSemicolonTags(values["quality_tags"]),
                tags: allTags.isEmpty ? parseSemicolonTags(values["entity_tags"]) : allTags
            )
        }
        .sorted { $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending }
    }

    private func loadAnalysisArtifacts(libraryURL: URL) throws -> [AnalysisArtifact] {
        try Self.analysisFiles.compactMap { artifact in
            let url = libraryURL.appendingPathComponent(artifact.filename)
            guard FileManager.default.fileExists(atPath: url.path) else {
                return nil
            }
            let content = try String(contentsOf: url, encoding: .utf8)
            return AnalysisArtifact(
                id: artifact.filename,
                title: firstMarkdownHeading(in: content) ?? artifact.fallbackTitle,
                filename: artifact.filename,
                content: content
            )
        }
    }

    private func loadPeople(libraryURL: URL, files: [TranscriptFile]) throws -> [PersonProfile] {
        let appearances = try loadPersonAppearances(libraryURL: libraryURL, files: files)
        let tags = try loadPersonTags(libraryURL: libraryURL)
        let knownPersonIDs = Set(appearances.map(\.personID)).union(tags.keys)
        return knownPersonIDs.map { personID in
            let tagged = tags[personID]
            return PersonProfile(
                id: personID,
                displayName: tagged?.displayName ?? "",
                tags: tagged?.tags ?? [],
                notes: tagged?.notes ?? "",
                appearances: appearances.filter { $0.personID == personID }.sorted {
                    if $0.relativePath == $1.relativePath {
                        return $0.timestamp < $1.timestamp
                    }
                    return $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending
                }
            )
        }
        .sorted {
            if $0.videoCount != $1.videoCount {
                return $0.videoCount > $1.videoCount
            }
            return $0.title.localizedStandardCompare($1.title) == .orderedAscending
        }
    }

    private func loadPersonAppearances(libraryURL: URL, files: [TranscriptFile]) throws -> [PersonAppearance] {
        let url = libraryURL.appendingPathComponent(Self.peopleIndexFilename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }

        let sourceByRelativePath = Dictionary(uniqueKeysWithValues: files.map { ($0.relativePath, $0.sourceURL) })
        let fileIDByRelativePath = Dictionary(uniqueKeysWithValues: files.map { ($0.relativePath, $0.id) })
        let text = try String(contentsOf: url, encoding: .utf8)
        let rows = CSV.parse(text).filter { !$0.allSatisfy(\.isEmpty) }
        guard let header = rows.first else { return [] }
        return rows.dropFirst().compactMap { row in
            let values = Dictionary(uniqueKeysWithValues: header.enumerated().map { index, column in
                (column, index < row.count ? row[index] : "")
            })
            guard
                let personID = nonEmpty(values["person_id"]),
                let relativePath = nonEmpty(values["file"]),
                let timestamp = Double(values["timestamp"] ?? ""),
                let x = Double(values["bbox_x"] ?? ""),
                let y = Double(values["bbox_y"] ?? ""),
                let width = Double(values["bbox_width"] ?? ""),
                let height = Double(values["bbox_height"] ?? "")
            else {
                return nil
            }
            let fileID = nonEmpty(values["file_id"]) ?? fileIDByRelativePath[relativePath] ?? stableFileID(sourceByRelativePath[relativePath]?.path ?? relativePath)
            let appearanceID = nonEmpty(values["appearance_id"]) ?? stableID(personID, relativePath, timestamp, x, y, width, height)
            return PersonAppearance(
                id: appearanceID,
                personID: personID,
                fileID: fileID,
                relativePath: relativePath,
                sourceURL: sourceByRelativePath[relativePath],
                timestamp: timestamp,
                boundingBox: FaceBoundingBox(x: x, y: y, width: width, height: height),
                signature: values["signature"] ?? ""
            )
        }
    }

    private func loadPersonTags(libraryURL: URL) throws -> [String: (displayName: String, tags: [String], notes: String)] {
        let url = libraryURL.appendingPathComponent(Self.peopleTagsFilename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return [:]
        }

        let text = try String(contentsOf: url, encoding: .utf8)
        let rows = CSV.parse(text).filter { !$0.allSatisfy(\.isEmpty) }
        guard let header = rows.first else { return [:] }
        return Dictionary(uniqueKeysWithValues: rows.dropFirst().compactMap { row in
            let values = Dictionary(uniqueKeysWithValues: header.enumerated().map { index, column in
                (column, index < row.count ? row[index] : "")
            })
            guard let personID = nonEmpty(values["person_id"]) else { return nil }
            let tags = (values["tags"] ?? "")
                .split(separator: ",")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            return (personID, (values["display_name"] ?? "", tags, values["notes"] ?? ""))
        })
    }

    private func readTranscript(jsonURL: URL) throws -> WhisperXTranscript {
        do {
            let decoder = JSONDecoder()
            return try decoder.decode(WhisperXTranscript.self, from: Data(contentsOf: jsonURL))
        } catch {
            throw LibraryStoreError.unreadableTranscript(jsonURL)
        }
    }

    private func requiredValue(_ key: String, in row: [String: String]) throws -> String {
        guard let value = row[key], !value.isEmpty else {
            throw LibraryStoreError.missingColumn(key)
        }
        return value
    }

    private func stableFileID(_ path: String) -> String {
        stableID(path)
    }

    private func stableID(_ parts: Any...) -> String {
        var hash: UInt64 = 1469598103934665603
        for part in parts {
            for byte in String(describing: part).utf8 {
                hash ^= UInt64(byte)
                hash &*= 1099511628211
            }
            hash ^= 31
            hash &*= 1099511628211
        }
        return String(hash, radix: 16)
    }

    private func parseTimecode(_ text: String) -> Double? {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let seconds = Double(value) {
            return seconds
        }
        let parts = value.split(separator: ":").compactMap { Double($0) }
        switch parts.count {
        case 2:
            return parts[0] * 60 + parts[1]
        case 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        default:
            return nil
        }
    }

    private func firstMarkdownHeading(in content: String) -> String? {
        content
            .split(whereSeparator: \.isNewline)
            .first { $0.hasPrefix("# ") }
            .map { String($0.dropFirst(2)).trimmingCharacters(in: .whitespacesAndNewlines) }
            .flatMap(nonEmpty)
    }

    private func nonEmpty(_ value: String?) -> String? {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private func parseSemicolonTags(_ value: String?) -> [String] {
        (value ?? "")
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func tagValue(for column: String, person: PersonProfile) -> String {
        switch column {
        case "person_id":
            person.id
        case "display_name":
            person.displayName
        case "tags":
            person.tags.joined(separator: ", ")
        case "notes":
            person.notes
        case "updated_at":
            Self.formatNumber(Date().timeIntervalSince1970)
        default:
            ""
        }
    }

    private static func formatNumber(_ value: Double) -> String {
        String(format: "%.6f", value)
    }
}

private struct WhisperXTranscript: Decodable {
    var language: String?
    var segments: [WhisperXSegment]
}

private struct WhisperXSegment: Decodable {
    var start: Double?
    var end: Double?
    var speaker: String?
    var text: String
    var averageLogProbability: Double?

    enum CodingKeys: String, CodingKey {
        case start
        case end
        case speaker
        case text
        case averageLogProbability = "avg_logprob"
    }
}
