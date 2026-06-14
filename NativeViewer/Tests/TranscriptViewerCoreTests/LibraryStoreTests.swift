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
        #expect(snapshot.clipMoments.first?.quality == "good")
        #expect(snapshot.clipMoments.first?.sourceURL == mediaURL(in: fixture))
        #expect(snapshot.clipTags.count == 1)
        #expect(snapshot.clipTags.first?.relativePath == "Day 1/C0001.MP4")
        #expect(snapshot.clipTags.first?.locationTags == ["Hangzhou", "Zhejiang University"])
        #expect(snapshot.clipTags.first?.spokenLanguageTags == ["Chinese"])
        #expect(snapshot.clipTags.first?.themeTags == ["Education and learning", "Everyday utility"])
        #expect(snapshot.clipTags.first?.entityTags == ["Codex", "DeepSeek"])
        #expect(snapshot.clipTags.first?.interviewLanguageTags == ["Computer-Using AI"])
        #expect(snapshot.clipTags.first?.qualityTags == ["very_short"])
        #expect(snapshot.clipTags.first?.tags.contains("Codex") == true)
        #expect(snapshot.people.count == 1)
        #expect(snapshot.people.first?.title == "Host")
        #expect(snapshot.people.first?.tags == ["host", "decision maker"])
        #expect(snapshot.people.first?.videoCount == 1)
        #expect(snapshot.people.first?.appearances.first?.relativePath == "Day 1/C0001.MP4")
        #expect(snapshot.analysisArtifacts.map(\.filename).contains("best_50_quote_moments.md"))
        #expect(snapshot.analysisArtifacts.first(where: { $0.filename == "best_50_quote_moments.md" })?.title == "Best 50 Quote Moments")
    }

    @Test("saves editable people tags")
    func savesPeopleTags() throws {
        let fixture = try Fixture()
        let store = LibraryStore()
        let snapshot = try store.load(libraryURL: fixture.libraryURL)
        let person = try #require(snapshot.people.first)

        try store.savePersonTags(
            libraryURL: fixture.libraryURL,
            person: PersonProfile(
                id: person.id,
                displayName: "Founder",
                tags: ["investor", "clip priority"],
                notes: "Use for product narrative.",
                appearances: person.appearances
            )
        )

        let updated = try store.load(libraryURL: fixture.libraryURL).people.first
        #expect(updated?.title == "Founder")
        #expect(updated?.tags == ["investor", "clip priority"])
        #expect(updated?.notes == "Use for product narrative.")
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
            ["file", "start_time", "end_time", "theme", "hook_strength", "quality", "speaker", "text"],
            ["Day 1/C0001.MP4", "00:05", "00:08.75", "AI lacks taste", "high", "good", "SPEAKER_01", "但是味道还是人来判断"]
        ]
        try CSV.encode(rows: clipMomentRows)
            .write(to: libraryURL.appendingPathComponent("clip_moments.csv"), atomically: true, encoding: .utf8)

        let clipTagRows = [
            [
                "file",
                "relative_file",
                "json",
                "status",
                "segment_count",
                "location_tags",
                "spoken_language_tags",
                "theme_tags",
                "entity_tags",
                "interview_language_tags",
                "quality_tags",
                "tags",
                "text_excerpt"
            ],
            [
                mediaURL.path,
                "Day 1/C0001.MP4",
                transcriptURL.path,
                "done",
                "2",
                "Hangzhou;Zhejiang University",
                "Chinese",
                "Education and learning;Everyday utility",
                "Codex;DeepSeek",
                "Computer-Using AI",
                "very_short",
                "Hangzhou;Zhejiang University;Chinese;Education and learning;Everyday utility;Codex;DeepSeek;Computer-Using AI;very_short",
                "AI 可以帮我做基础工作"
            ]
        ]
        try CSV.encode(rows: clipTagRows)
            .write(to: libraryURL.appendingPathComponent("clip_tags.csv"), atomically: true, encoding: .utf8)

        let peopleRows = [
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
            ],
            ["person_abc123", "appearance_1", "Day 1/C0001.MP4", "", "5.500", "0.2", "0.2", "0.3", "0.3", "0.1;0.2;0.3"]
        ]
        try CSV.encode(rows: peopleRows)
            .write(to: libraryURL.appendingPathComponent("people_index.csv"), atomically: true, encoding: .utf8)

        let peopleTagRows = [
            ["person_id", "display_name", "tags", "notes", "updated_at"],
            ["person_abc123", "Host", "host, decision maker", "Primary speaker", "0"]
        ]
        try CSV.encode(rows: peopleTagRows)
            .write(to: libraryURL.appendingPathComponent("people_tags.csv"), atomically: true, encoding: .utf8)

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
