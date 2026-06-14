import Foundation
import Testing
@testable import TranscriptViewerCore

@Suite("Transcript viewer core")
struct LibraryStoreTests {
    @Test("loads manifest and WhisperX transcript segments")
    func loadsLibrary() throws {
        let fixture = try Fixture()
        let snapshot = try LibraryStore().load(libraryURL: fixture.libraryURL)

        #expect(snapshot.files.count == 2)
        #expect(snapshot.doneFileCount == 1)
        #expect(snapshot.pendingFileCount == 1)
        #expect(snapshot.segments.count == 2)
        #expect(snapshot.segments.first?.speaker == "SPEAKER_00")
        #expect(snapshot.segments.first?.language == "zh")
        #expect(snapshot.segments.first?.text == "AI 可以帮我做基础工作")
        #expect(snapshot.clipMoments.count == 1)
        #expect(snapshot.clipMoments.first?.start == 5)
        #expect(snapshot.clipMoments.first?.end == 8.75)
        #expect(snapshot.clipMoments.first?.theme == "AI lacks taste")
        #expect(snapshot.clipMoments.first?.sourceURL == mediaURL(in: fixture))
        #expect(snapshot.analysisArtifacts.map(\.filename).contains("best_50_quote_moments.md"))
        #expect(snapshot.analysisArtifacts.first(where: { $0.filename == "best_50_quote_moments.md" })?.title == "Best 50 Quote Moments")
    }

    @Test("persists selects and exports stable CSV")
    func savesAndExportsSelects() throws {
        let fixture = try Fixture()
        let store = LibraryStore()
        let snapshot = try store.load(libraryURL: fixture.libraryURL)
        let segment = try #require(snapshot.segments.last)
        let select = SelectMoment(
            id: "select::\(segment.id)",
            segmentID: segment.id,
            fileID: segment.fileID,
            sourceURL: segment.sourceURL,
            relativePath: segment.relativePath,
            start: 5.25,
            end: 8.5,
            speaker: segment.speaker,
            text: segment.text,
            status: .good,
            hookStrength: 5,
            theme: "AI lacks taste",
            tags: ["food", "human judgment"],
            notes: "Strong contrast",
            createdAt: Date(timeIntervalSince1970: 10),
            updatedAt: Date(timeIntervalSince1970: 20)
        )

        try store.save(selects: [select], libraryURL: fixture.libraryURL)
        let reloaded = try store.load(libraryURL: fixture.libraryURL)
        #expect(reloaded.selects == [select])

        let exportURL = try store.export(selects: reloaded.selects, libraryURL: fixture.libraryURL)
        let csv = try String(contentsOf: exportURL, encoding: .utf8)
        #expect(csv.contains("file,relative_file,start,end,theme,hook_strength,status,tags,speaker,text,notes,segment_id"))
        #expect(csv.contains("Day 1/C0001.MP4"))
        #expect(csv.contains("5.250,8.500"))
        #expect(csv.contains("\"food,human judgment\""))
    }

    @Test("CSV parser handles quoted commas and escaped quotes")
    func parsesQuotedCSV() {
        let rows = CSV.parse("a,b,c\none,\"two, still two\",\"quote \"\"inside\"\"\"\n")
        #expect(rows == [["a", "b", "c"], ["one", "two, still two", "quote \"inside\""]])
    }

    @Test("CSV parser handles CRLF rows")
    func parsesCRLFCSV() {
        let rows = CSV.parse("file,relative_file,output_dir,json,status\r\none,two,three,four,done\r\n")
        #expect(rows == [["file", "relative_file", "output_dir", "json", "status"], ["one", "two", "three", "four", "done"]])
    }
}

private struct Fixture {
    let rootURL: URL
    let libraryURL: URL

    init() throws {
        rootURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("TranscriptViewerTests-\(UUID().uuidString)", isDirectory: true)
        libraryURL = rootURL.appendingPathComponent("_ai_library", isDirectory: true)
        let transcriptDirectory = libraryURL
            .appendingPathComponent("transcripts", isDirectory: true)
            .appendingPathComponent("Day 1", isDirectory: true)
            .appendingPathComponent("C0001", isDirectory: true)
        let mediaDirectory = rootURL.appendingPathComponent("Day 1", isDirectory: true)
        try FileManager.default.createDirectory(at: transcriptDirectory, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: mediaDirectory, withIntermediateDirectories: true)

        let mediaURL = mediaDirectory.appendingPathComponent("C0001.MP4")
        try Data("fake-video".utf8).write(to: mediaURL)

        let transcriptURL = transcriptDirectory.appendingPathComponent("C0001.json")
        try """
        {
          "language": "zh",
          "segments": [
            {
              "start": 1.25,
              "end": 4.5,
              "speaker": "SPEAKER_00",
              "text": "AI 可以帮我做基础工作",
              "avg_logprob": -0.2
            },
            {
              "start": 5.0,
              "end": 8.75,
              "text": "但是味道还是人来判断"
            }
          ]
        }
        """.write(to: transcriptURL, atomically: true, encoding: .utf8)

        let pendingURL = mediaDirectory.appendingPathComponent("C0002.MP4")
        let manifestRows = [
            ["file", "relative_file", "output_dir", "json", "status"],
            [mediaURL.path, "Day 1/C0001.MP4", transcriptDirectory.path, transcriptURL.path, "done"],
            [
                pendingURL.path,
                "Day 1/C0002.MP4",
                libraryURL.appendingPathComponent("transcripts/Day 1/C0002").path,
                libraryURL.appendingPathComponent("transcripts/Day 1/C0002/C0002.json").path,
                "pending"
            ]
        ]
        try CSV.encode(rows: manifestRows)
            .write(to: libraryURL.appendingPathComponent("manifest.csv"), atomically: true, encoding: .utf8)

        let clipMomentRows = [
            ["file", "start_time", "end_time", "theme", "hook_strength", "speaker", "text"],
            ["Day 1/C0001.MP4", "00:05", "00:08.75", "AI lacks taste", "high", "SPEAKER_01", "但是味道还是人来判断"]
        ]
        try CSV.encode(rows: clipMomentRows)
            .write(to: libraryURL.appendingPathComponent("clip_moments.csv"), atomically: true, encoding: .utf8)

        try """
        # Best 50 Quote Moments

        1. `Day 1/C0001.MP4` 00:05-00:08 `SPEAKER_01` [AI lacks taste, high]
           但是味道还是人来判断
        """.write(to: libraryURL.appendingPathComponent("best_50_quote_moments.md"), atomically: true, encoding: .utf8)
    }
}

private func mediaURL(in fixture: Fixture) -> URL {
    fixture.rootURL
        .appendingPathComponent("Day 1", isDirectory: true)
        .appendingPathComponent("C0001.MP4")
        .standardizedFileURL
}
