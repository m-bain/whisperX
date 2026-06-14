// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "WhisperXTranscriptViewer",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "TranscriptViewer", targets: ["TranscriptViewer"])
    ],
    targets: [
        .target(name: "TranscriptViewerCore"),
        .executableTarget(
            name: "TranscriptViewer",
            dependencies: ["TranscriptViewerCore"],
            linkerSettings: [
                .linkedFramework("AVFoundation"),
                .linkedFramework("AVKit"),
                .linkedFramework("Vision")
            ]
        ),
        .testTarget(
            name: "TranscriptViewerCoreTests",
            dependencies: ["TranscriptViewerCore"]
        )
    ]
)
