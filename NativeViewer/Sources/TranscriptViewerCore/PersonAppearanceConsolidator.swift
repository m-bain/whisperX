import Foundation

public struct PersonAppearanceConsolidationResult: Sendable {
    public var appearances: [PersonAppearance]
    public var personIDByOriginalID: [String: String]

    public init(appearances: [PersonAppearance], personIDByOriginalID: [String: String]) {
        self.appearances = appearances
        self.personIDByOriginalID = personIDByOriginalID
    }
}

public struct PersonAppearanceConsolidator: Sendable {
    public var mergeThreshold: Double

    public init(mergeThreshold: Double = 0.24) {
        self.mergeThreshold = mergeThreshold
    }

    public func consolidate(_ appearances: [PersonAppearance]) -> [PersonAppearance] {
        consolidateWithRemapping(appearances).appearances
    }

    public func consolidateWithRemapping(_ appearances: [PersonAppearance]) -> PersonAppearanceConsolidationResult {
        var clustersByID: [String: SignatureCluster] = [:]
        var originalIDs = Set(appearances.map(\.personID))

        for appearance in appearances {
            guard let values = Self.signatureValues(from: appearance.signature) else {
                continue
            }
            originalIDs.insert(appearance.personID)
            clustersByID[appearance.personID, default: SignatureCluster(id: appearance.personID)].add(values)
        }

        var clusters = clustersByID.values.sorted { $0.id < $1.id }
        while let pair = closestMergeablePair(in: clusters) {
            let merged = clusters[pair.left].merged(with: clusters[pair.right])
            clusters.remove(at: pair.right)
            clusters.remove(at: pair.left)
            clusters.append(merged)
            clusters.sort { $0.id < $1.id }
        }

        var personIDByOriginalID = Dictionary(uniqueKeysWithValues: originalIDs.map { ($0, $0) })
        for cluster in clusters {
            for memberID in cluster.memberIDs {
                personIDByOriginalID[memberID] = cluster.id
            }
        }

        let consolidatedAppearances = appearances.map { appearance in
            var updated = appearance
            updated.personID = personIDByOriginalID[appearance.personID] ?? appearance.personID
            return updated
        }

        return PersonAppearanceConsolidationResult(
            appearances: consolidatedAppearances,
            personIDByOriginalID: personIDByOriginalID
        )
    }

    private func closestMergeablePair(in clusters: [SignatureCluster]) -> (left: Int, right: Int)? {
        var best: (left: Int, right: Int, distance: Double)?
        for left in clusters.indices {
            for right in clusters.indices.dropFirst(left + 1) {
                let distance = Self.rootMeanSquareDistance(clusters[left].centroid, clusters[right].centroid)
                guard distance <= mergeThreshold else { continue }
                if best == nil || distance < best!.distance || (distance == best!.distance && clusters[left].id < clusters[best!.left].id) {
                    best = (left, right, distance)
                }
            }
        }
        guard let best else { return nil }
        return (best.left, best.right)
    }

    private static func signatureValues(from signature: String) -> [Double]? {
        let components = signature
            .split(separator: ";")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let values = components.compactMap(Double.init)
        guard values.count == components.count else { return nil }
        return values.isEmpty ? nil : values
    }

    private static func rootMeanSquareDistance(_ lhs: [Double], _ rhs: [Double]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return .greatestFiniteMagnitude }
        let squared = zip(lhs, rhs).map { pow($0 - $1, 2) }.reduce(0, +)
        return sqrt(squared / Double(lhs.count))
    }
}

private struct SignatureCluster {
    var id: String
    var centroid: [Double] = []
    var signatureCount = 0
    var appearanceCount = 0
    var memberIDs: Set<String> = []

    mutating func add(_ signature: [Double]) {
        guard centroid.isEmpty || centroid.count == signature.count else { return }
        memberIDs.insert(id)
        appearanceCount += 1
        let nextCount = signatureCount + 1
        if centroid.isEmpty {
            centroid = signature
        } else {
            centroid = zip(centroid, signature).map { (($0 * Double(signatureCount)) + $1) / Double(nextCount) }
        }
        signatureCount = nextCount
    }

    func merged(with other: SignatureCluster) -> SignatureCluster {
        let totalSignatures = signatureCount + other.signatureCount
        let mergedCentroid = zip(centroid, other.centroid).map {
            (($0 * Double(signatureCount)) + ($1 * Double(other.signatureCount))) / Double(totalSignatures)
        }
        return SignatureCluster(
            id: preferredID(over: other),
            centroid: mergedCentroid,
            signatureCount: totalSignatures,
            appearanceCount: appearanceCount + other.appearanceCount,
            memberIDs: memberIDs.union(other.memberIDs)
        )
    }

    private func preferredID(over other: SignatureCluster) -> String {
        if appearanceCount != other.appearanceCount {
            return appearanceCount > other.appearanceCount ? id : other.id
        }
        return min(id, other.id)
    }
}
