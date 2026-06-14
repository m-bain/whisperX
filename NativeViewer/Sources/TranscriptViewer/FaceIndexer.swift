import AVFoundation
import AppKit
import Foundation
import TranscriptViewerCore
import Vision

struct FaceIndexer: Sendable {
    var sampleInterval: Double = 5
    var maximumFramesPerFile: Int = 180
    var clusterThreshold: Double = 0.18

    func scan(files: [TranscriptFile]) async throws -> [PersonAppearance] {
        try await Task.detached(priority: .userInitiated) {
            try await scanSynchronously(files: files)
        }.value
    }

    private func scanSynchronously(files: [TranscriptFile]) async throws -> [PersonAppearance] {
        var clusters: [FaceCluster] = []
        var appearances: [PersonAppearance] = []

        for file in files where file.status == "done" {
            let asset = AVURLAsset(url: file.sourceURL)
            let duration = try await asset.load(.duration).seconds
            guard duration.isFinite, duration >= 0 else { continue }

            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.requestedTimeToleranceBefore = .zero
            generator.requestedTimeToleranceAfter = .zero

            for timestamp in sampleTimes(duration: duration) {
                guard let image = try? generator.copyCGImage(at: CMTime(seconds: timestamp, preferredTimescale: 600), actualTime: nil) else {
                    continue
                }
                let detections = try detectFaces(in: image)
                for detection in detections {
                    guard let signature = signature(for: detection.boundingBox, in: image) else {
                        continue
                    }
                    let personID: String
                    if let matchIndex = clusters.bestMatch(for: signature.values, threshold: clusterThreshold) {
                        clusters[matchIndex].add(signature.values)
                        personID = clusters[matchIndex].id
                    } else {
                        personID = "person_\(stableID(signature.encoded))"
                        clusters.append(FaceCluster(id: personID, centroid: signature.values))
                    }
                    let box = FaceBoundingBox(
                        x: detection.boundingBox.minX,
                        y: detection.boundingBox.minY,
                        width: detection.boundingBox.width,
                        height: detection.boundingBox.height
                    )
                    appearances.append(
                        PersonAppearance(
                            id: stableID(file.id, timestamp, box.x, box.y, box.width, box.height),
                            personID: personID,
                            fileID: file.id,
                            relativePath: file.relativePath,
                            sourceURL: file.sourceURL,
                            timestamp: timestamp,
                            boundingBox: box,
                            signature: signature.encoded
                        )
                    )
                }
            }
        }

        return appearances
    }

    private func sampleTimes(duration: Double) -> [Double] {
        let frameCount = min(maximumFramesPerFile, max(1, Int(ceil(duration / sampleInterval)) + 1))
        guard frameCount > 1 else { return [0] }
        let step = max(sampleInterval, duration / Double(frameCount - 1))
        return (0..<frameCount).map { min(duration, Double($0) * step) }
    }

    private func detectFaces(in image: CGImage) throws -> [VNFaceObservation] {
        let request = VNDetectFaceRectanglesRequest()
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])
        return request.results ?? []
    }

    private func signature(for boundingBox: CGRect, in image: CGImage) -> FaceSignature? {
        let width = CGFloat(image.width)
        let height = CGFloat(image.height)
        let cropRect = CGRect(
            x: boundingBox.minX * width,
            y: (1 - boundingBox.maxY) * height,
            width: boundingBox.width * width,
            height: boundingBox.height * height
        ).integral.intersection(CGRect(x: 0, y: 0, width: width, height: height))
        guard cropRect.width > 8, cropRect.height > 8, let crop = image.cropping(to: cropRect) else {
            return nil
        }

        let bitmap = NSBitmapImageRep(cgImage: crop)
        let grid = 12
        var values: [Double] = []
        values.reserveCapacity(grid * grid)
        for row in 0..<grid {
            for column in 0..<grid {
                let x = min(bitmap.pixelsWide - 1, max(0, Int((Double(column) + 0.5) * Double(bitmap.pixelsWide) / Double(grid))))
                let y = min(bitmap.pixelsHigh - 1, max(0, Int((Double(row) + 0.5) * Double(bitmap.pixelsHigh) / Double(grid))))
                let color = bitmap.colorAt(x: x, y: y) ?? .black
                values.append((0.299 * color.redComponent) + (0.587 * color.greenComponent) + (0.114 * color.blueComponent))
            }
        }
        let average = values.reduce(0, +) / Double(values.count)
        let normalized = values.map { min(1, max(0, ($0 - average) + 0.5)) }
        let encoded = normalized.map { String(format: "%.3f", $0) }.joined(separator: ";")
        return FaceSignature(values: normalized, encoded: encoded)
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
}

private struct FaceSignature {
    var values: [Double]
    var encoded: String
}

private struct FaceCluster {
    var id: String
    var centroid: [Double]
    var count = 1

    mutating func add(_ signature: [Double]) {
        let nextCount = count + 1
        centroid = zip(centroid, signature).map { (($0 * Double(count)) + $1) / Double(nextCount) }
        count = nextCount
    }
}

private extension Array where Element == FaceCluster {
    func bestMatch(for signature: [Double], threshold: Double) -> Int? {
        enumerated()
            .map { index, cluster in (index, rootMeanSquareDistance(signature, cluster.centroid)) }
            .filter { $0.1 <= threshold }
            .min { $0.1 < $1.1 }?
            .0
    }

    private func rootMeanSquareDistance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return .greatestFiniteMagnitude }
        let squared = zip(lhs, rhs).map { pow($0 - $1, 2) }.reduce(0, +)
        return sqrt(squared / Double(lhs.count))
    }
}
