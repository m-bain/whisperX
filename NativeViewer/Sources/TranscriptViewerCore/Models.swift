import Foundation

public struct LibrarySnapshot: Equatable, Sendable {
    public var libraryURL: URL
    public var files: [TranscriptFile]
    public var segments: [TranscriptSegment]
    public var clipMoments: [ClipMoment]
    public var analysisArtifacts: [AnalysisArtifact]

    public init(
        libraryURL: URL,
        files: [TranscriptFile],
        segments: [TranscriptSegment],
        clipMoments: [ClipMoment] = [],
        analysisArtifacts: [AnalysisArtifact] = []
    ) {
        self.libraryURL = libraryURL
        self.files = files
        self.segments = segments
        self.clipMoments = clipMoments
        self.analysisArtifacts = analysisArtifacts
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

public struct ClipMoment: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var relativePath: String
    public var sourceURL: URL?
    public var start: Double
    public var end: Double
    public var theme: String
    public var hookStrength: String
    public var quality: String
    public var speaker: String?
    public var text: String

    public init(
        id: String,
        relativePath: String,
        sourceURL: URL?,
        start: Double,
        end: Double,
        theme: String,
        hookStrength: String,
        quality: String,
        speaker: String?,
        text: String
    ) {
        self.id = id
        self.relativePath = relativePath
        self.sourceURL = sourceURL
        self.start = start
        self.end = end
        self.theme = theme
        self.hookStrength = hookStrength
        self.quality = quality
        self.speaker = speaker
        self.text = text
    }
}

public struct AnalysisArtifact: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var title: String
    public var filename: String
    public var content: String

    public init(id: String, title: String, filename: String, content: String) {
        self.id = id
        self.title = title
        self.filename = filename
        self.content = content
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
