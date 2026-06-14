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
    public static let selectsFilename = "native-selects.json"
    public static let exportFilename = "selects.csv"

    public init() {}

    public func load(libraryURL: URL) throws -> LibrarySnapshot {
        let libraryURL = libraryURL.standardizedFileURL
        let filesAndSegments = try loadFilesAndSegments(libraryURL: libraryURL)
        let selects = try loadSelects(libraryURL: libraryURL)
        return LibrarySnapshot(
            libraryURL: libraryURL,
            files: filesAndSegments.files,
            segments: filesAndSegments.segments,
            selects: selects
        )
    }

    public func save(selects: [SelectMoment], libraryURL: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(selects.sorted { $0.updatedAt > $1.updatedAt })
        try data.write(to: libraryURL.appendingPathComponent(Self.selectsFilename), options: .atomic)
    }

    public func export(selects: [SelectMoment], libraryURL: URL) throws -> URL {
        let sorted = selects.sorted {
            if $0.relativePath == $1.relativePath {
                return $0.start < $1.start
            }
            return $0.relativePath < $1.relativePath
        }
        var rows = [[
            "file",
            "relative_file",
            "start",
            "end",
            "theme",
            "hook_strength",
            "status",
            "tags",
            "speaker",
            "text",
            "notes",
            "segment_id"
        ]]
        rows.append(contentsOf: sorted.map { select in
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
        })
        let url = libraryURL.appendingPathComponent(Self.exportFilename)
        try CSV.encode(rows: rows).write(to: url, atomically: true, encoding: .utf8)
        return url
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

    private func loadSelects(libraryURL: URL) throws -> [SelectMoment] {
        let url = libraryURL.appendingPathComponent(Self.selectsFilename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return []
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode([SelectMoment].self, from: Data(contentsOf: url))
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
        var hash: UInt64 = 1469598103934665603
        for byte in path.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        return String(hash, radix: 16)
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

