import Foundation

public struct LibrarySnapshot: Equatable, Sendable {
    public var libraryURL: URL
    public var files: [TranscriptFile]
    public var segments: [TranscriptSegment]
    public var clipMoments: [ClipMoment]
    public var clipTags: [ClipTag]
    public var analysisArtifacts: [AnalysisArtifact]
    public var people: [PersonProfile]

    public init(
        libraryURL: URL,
        files: [TranscriptFile],
        segments: [TranscriptSegment],
        clipMoments: [ClipMoment] = [],
        clipTags: [ClipTag] = [],
        analysisArtifacts: [AnalysisArtifact] = [],
        people: [PersonProfile] = []
    ) {
        self.libraryURL = libraryURL
        self.files = files
        self.segments = segments
        self.clipMoments = clipMoments
        self.clipTags = clipTags
        self.analysisArtifacts = analysisArtifacts
        self.people = people
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

public struct PersonProfile: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var displayName: String
    public var tags: [String]
    public var notes: String
    public var appearances: [PersonAppearance]

    public init(
        id: String,
        displayName: String = "",
        tags: [String] = [],
        notes: String = "",
        appearances: [PersonAppearance] = []
    ) {
        self.id = id
        self.displayName = displayName
        self.tags = tags
        self.notes = notes
        self.appearances = appearances
    }

    public var title: String {
        displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "Person \(id.suffix(6))" : displayName
    }

    public var videoCount: Int {
        Set(appearances.map(\.fileID)).count
    }

    public var appearanceCount: Int {
        appearances.count
    }

    public var fileIDs: Set<String> {
        Set(appearances.map(\.fileID))
    }
}

public struct PersonAppearance: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var personID: String
    public var fileID: String
    public var relativePath: String
    public var sourceURL: URL?
    public var timestamp: Double
    public var boundingBox: FaceBoundingBox
    public var signature: String

    public init(
        id: String,
        personID: String,
        fileID: String,
        relativePath: String,
        sourceURL: URL?,
        timestamp: Double,
        boundingBox: FaceBoundingBox,
        signature: String
    ) {
        self.id = id
        self.personID = personID
        self.fileID = fileID
        self.relativePath = relativePath
        self.sourceURL = sourceURL
        self.timestamp = timestamp
        self.boundingBox = boundingBox
        self.signature = signature
    }
}

public struct FaceBoundingBox: Hashable, Codable, Sendable {
    public var x: Double
    public var y: Double
    public var width: Double
    public var height: Double

    public init(x: Double, y: Double, width: Double, height: Double) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
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

public struct ClipTag: Identifiable, Hashable, Codable, Sendable {
    public var id: String
    public var relativePath: String
    public var locationTags: [String]
    public var spokenLanguageTags: [String]
    public var themeTags: [String]
    public var entityTags: [String]
    public var interviewLanguageTags: [String]
    public var qualityTags: [String]
    public var tags: [String]

    public init(
        id: String,
        relativePath: String,
        locationTags: [String] = [],
        spokenLanguageTags: [String] = [],
        themeTags: [String] = [],
        entityTags: [String] = [],
        interviewLanguageTags: [String] = [],
        qualityTags: [String] = [],
        tags: [String] = []
    ) {
        self.id = id
        self.relativePath = relativePath
        self.locationTags = locationTags
        self.spokenLanguageTags = spokenLanguageTags
        self.themeTags = themeTags
        self.entityTags = entityTags
        self.interviewLanguageTags = interviewLanguageTags
        self.qualityTags = qualityTags
        self.tags = tags
    }

    public var displayTags: [String] {
        var values: [String] = []
        appendUnique(locationTags, to: &values)
        appendUnique(entityTags, to: &values)
        appendUnique(interviewLanguageTags, to: &values)
        appendUnique(themeTags, to: &values)
        appendUnique(spokenLanguageTags, to: &values)
        appendUnique(qualityTags, to: &values)
        appendUnique(tags, to: &values)
        return values
    }

    private func appendUnique(_ tags: [String], to values: inout [String]) {
        for tag in tags where !values.contains(tag) {
            values.append(tag)
        }
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
