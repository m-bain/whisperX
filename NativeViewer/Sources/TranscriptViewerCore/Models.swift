import Foundation

public struct LibrarySnapshot: Equatable, Sendable {
    public var libraryURL: URL
    public var files: [TranscriptFile]
    public var segments: [TranscriptSegment]
    public var selects: [SelectMoment]

    public init(libraryURL: URL, files: [TranscriptFile], segments: [TranscriptSegment], selects: [SelectMoment]) {
        self.libraryURL = libraryURL
        self.files = files
        self.segments = segments
        self.selects = selects
    }

    public var doneFileCount: Int {
        files.filter { $0.status == "done" }.count
    }

    public var pendingFileCount: Int {
        files.filter { $0.status == "pending" }.count
    }

    public func segments(for fileID: String) -> [TranscriptSegment] {
        segments.filter { $0.fileID == fileID }
    }
}

public struct TranscriptFile: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var sourceURL: URL
    public var relativePath: String
    public var outputDirectory: URL
    public var jsonURL: URL
    public var status: String
    public var language: String?
    public var segmentCount: Int

    public init(
        id: String,
        sourceURL: URL,
        relativePath: String,
        outputDirectory: URL,
        jsonURL: URL,
        status: String,
        language: String?,
        segmentCount: Int
    ) {
        self.id = id
        self.sourceURL = sourceURL
        self.relativePath = relativePath
        self.outputDirectory = outputDirectory
        self.jsonURL = jsonURL
        self.status = status
        self.language = language
        self.segmentCount = segmentCount
    }
}

public struct TranscriptSegment: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var fileID: String
    public var sourceURL: URL
    public var relativePath: String
    public var index: Int
    public var start: Double
    public var end: Double
    public var speaker: String?
    public var text: String
    public var language: String?
    public var averageLogProbability: Double?

    public init(
        id: String,
        fileID: String,
        sourceURL: URL,
        relativePath: String,
        index: Int,
        start: Double,
        end: Double,
        speaker: String?,
        text: String,
        language: String?,
        averageLogProbability: Double?
    ) {
        self.id = id
        self.fileID = fileID
        self.sourceURL = sourceURL
        self.relativePath = relativePath
        self.index = index
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = text
        self.language = language
        self.averageLogProbability = averageLogProbability
    }
}

public struct SelectMoment: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var segmentID: String
    public var fileID: String
    public var sourceURL: URL
    public var relativePath: String
    public var start: Double
    public var end: Double
    public var speaker: String?
    public var text: String
    public var status: SelectStatus
    public var hookStrength: Int?
    public var theme: String
    public var tags: [String]
    public var notes: String
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: String,
        segmentID: String,
        fileID: String,
        sourceURL: URL,
        relativePath: String,
        start: Double,
        end: Double,
        speaker: String?,
        text: String,
        status: SelectStatus,
        hookStrength: Int?,
        theme: String,
        tags: [String],
        notes: String,
        createdAt: Date,
        updatedAt: Date
    ) {
        self.id = id
        self.segmentID = segmentID
        self.fileID = fileID
        self.sourceURL = sourceURL
        self.relativePath = relativePath
        self.start = start
        self.end = end
        self.speaker = speaker
        self.text = text
        self.status = status
        self.hookStrength = hookStrength
        self.theme = theme
        self.tags = tags
        self.notes = notes
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

public enum SelectStatus: String, CaseIterable, Codable, Sendable {
    case selected
    case good
    case maybe
    case weak
    case unusable

    public var title: String {
        switch self {
        case .selected: "Selected"
        case .good: "Good"
        case .maybe: "Maybe"
        case .weak: "Weak"
        case .unusable: "Unusable"
        }
    }
}

