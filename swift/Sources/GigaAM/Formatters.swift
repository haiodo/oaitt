// Subtitle and table formats, matching src/utils/formatters.py byte for byte.

import Foundation

private func timestamp(_ seconds: Double, separator: String) -> String {
    let hours = Int(seconds / 3600)
    let minutes = Int(seconds.truncatingRemainder(dividingBy: 3600) / 60)
    let secs = Int(seconds.truncatingRemainder(dividingBy: 60))
    let millis = Int(seconds.truncatingRemainder(dividingBy: 1) * 1000)
    return String(format: "%02d:%02d:%02d\(separator)%03d", hours, minutes, secs, millis)
}

public enum Formatters {
    public static func srt(_ segments: [Segment]) -> String {
        var lines: [String] = []
        for (index, s) in segments.enumerated() {
            lines.append(String(index + 1))
            lines.append(
                "\(timestamp(s.start, separator: ",")) --> \(timestamp(s.end, separator: ","))")
            lines.append(s.text)
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    public static func vtt(_ segments: [Segment]) -> String {
        var lines = ["WEBVTT", ""]
        for s in segments {
            lines.append(
                "\(timestamp(s.start, separator: ".")) --> \(timestamp(s.end, separator: "."))")
            lines.append(s.text)
            lines.append("")
        }
        return lines.joined(separator: "\n")
    }

    public static func tsv(_ segments: [Segment]) -> String {
        var lines = ["start\tend\ttext"]
        for s in segments {
            let text = s.text.replacingOccurrences(of: "\t", with: " ")
            lines.append("\(Int(s.start * 1000))\t\(Int(s.end * 1000))\t\(text)")
        }
        return lines.joined(separator: "\n")
    }
}

/// Quality signals the engine can produce. GigaAM has no word-level scores, so only the
/// character-rate heuristic from src/routes/openai.py:185-220 applies here.
public struct ConfidenceMetrics: Sendable {
    public var charsPerSecond: Double?
    public var charsPerSecondRatio: Double?
    public var charsPerSecondThreshold: Double
    public var highCharRate: Bool
    public var isReliable: Bool
    public var rejectionReasons: [String]

    public static func evaluate(
        text: String, audioSeconds: Double,
        maxCharsPerSecond: Double = 25.0, multiplier: Double = 3.0, minAudioSec: Double = 0.5
    ) -> ConfidenceMetrics {
        let threshold = maxCharsPerSecond * multiplier
        let rate = audioSeconds > 0 ? Double(text.count) / audioSeconds : nil
        let ratio = (maxCharsPerSecond > 0 ? rate.map { $0 / maxCharsPerSecond } : nil)

        var metrics = ConfidenceMetrics(
            charsPerSecond: rate.map { ($0 * 10000).rounded() / 10000 },
            charsPerSecondRatio: ratio.map { ($0 * 10000).rounded() / 10000 },
            charsPerSecondThreshold: (threshold * 10000).rounded() / 10000,
            highCharRate: false, isReliable: true, rejectionReasons: [])

        if let rate, let ratio, audioSeconds >= minAudioSec, maxCharsPerSecond > 0,
            ratio > multiplier
        {
            metrics.highCharRate = true
            metrics.isReliable = false
            metrics.rejectionReasons.append(
                String(format: "chars_per_second=%.2f > threshold=%.2f", rate, threshold))
        }
        return metrics
    }

    public var json: [String: Any] {
        var out: [String: Any] = [
            "chars_per_second_threshold": charsPerSecondThreshold,
            "high_char_rate": highCharRate,
            "is_reliable": isReliable,
        ]
        if let charsPerSecond { out["chars_per_second"] = charsPerSecond }
        if let charsPerSecondRatio { out["chars_per_second_ratio"] = charsPerSecondRatio }
        if !rejectionReasons.isEmpty { out["rejection_reasons"] = rejectionReasons }
        return out
    }
}
