// Minimal SentencePiece decoder: reads piece strings straight out of the
// .model protobuf. Encoding is not needed - ASR only ever decodes ids.

import Foundation

public struct SentencePieceTokenizer: Sendable {
    public let pieces: [String]

    public init(path: URL) throws {
        let data = try Data(contentsOf: path)
        var pieces: [String] = []
        var i = data.startIndex

        // ModelProto.pieces is field 1, wire type 2 (length-delimited message).
        while i < data.endIndex {
            guard let (key, next) = Self.varint(data, i) else { break }
            i = next
            let (field, wire) = (key >> 3, key & 0x7)
            guard let (len, afterLen) = Self.payloadLength(data, i, wire: wire) else { break }
            let end = data.index(afterLen, offsetBy: len, limitedBy: data.endIndex) ?? data.endIndex
            if field == 1 && wire == 2, let piece = Self.readPiece(data[afterLen..<end]) {
                pieces.append(piece)
            }
            i = end
        }

        guard !pieces.isEmpty else {
            throw NSError(
                domain: "SentencePiece", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "no pieces found in \(path.path)"])
        }
        self.pieces = pieces
    }

    public func decode(_ ids: [Int]) -> String {
        let raw = ids.compactMap { $0 < pieces.count ? pieces[$0] : nil }.joined()
        return raw.replacingOccurrences(of: "\u{2581}", with: " ")
            .trimmingCharacters(in: .whitespaces)
    }

    /// SentencePiece.piece is field 1, wire type 2.
    private static func readPiece(_ body: Data) -> String? {
        var i = body.startIndex
        while i < body.endIndex {
            guard let (key, next) = varint(body, i) else { return nil }
            i = next
            let (field, wire) = (key >> 3, key & 0x7)
            guard let (len, afterLen) = payloadLength(body, i, wire: wire) else { return nil }
            let end = body.index(afterLen, offsetBy: len, limitedBy: body.endIndex) ?? body.endIndex
            if field == 1 && wire == 2 {
                return String(data: body[afterLen..<end], encoding: .utf8)
            }
            i = end
        }
        return nil
    }

    /// Byte length of a field's payload plus the index where it starts.
    private static func payloadLength(
        _ d: Data, _ i: Data.Index, wire: UInt64
    ) -> (Int, Data.Index)? {
        switch wire {
        case 0: return varint(d, i).map { (0, $0.1) }
        case 1: return (8, i)
        case 2: return varint(d, i).map { (Int($0.0), $0.1) }
        case 5: return (4, i)
        default: return nil
        }
    }

    private static func varint(_ d: Data, _ start: Data.Index) -> (UInt64, Data.Index)? {
        var value: UInt64 = 0
        var shift: UInt64 = 0
        var i = start
        while i < d.endIndex {
            let byte = d[i]
            value |= UInt64(byte & 0x7F) << shift
            i = d.index(after: i)
            if byte & 0x80 == 0 { return (value, i) }
            shift += 7
            if shift > 63 { return nil }
        }
        return nil
    }
}
