// ponytail: minimal multipart/form-data reader - enough for OpenAI-style clients
// (one file part + flat text fields). Swap for MultipartKit if nested parts appear.

import Foundation

struct MultipartPart {
    let name: String
    let filename: String?
    let body: [UInt8]
}

enum Multipart {
    static func boundary(fromContentType contentType: String) -> String? {
        guard let range = contentType.range(of: "boundary=") else { return nil }
        var value = String(contentType[range.upperBound...])
        if let semicolon = value.firstIndex(of: ";") { value = String(value[..<semicolon]) }
        return value.trimmingCharacters(in: CharacterSet(charactersIn: "\" "))
    }

    static func parse(_ bytes: [UInt8], boundary: String) -> [MultipartPart] {
        let delimiter = Array("--\(boundary)".utf8)
        let separator = Array("\r\n\r\n".utf8)
        let starts = search(delimiter, in: bytes, from: 0, to: bytes.count)
        guard starts.count >= 2 else { return [] }

        var parts: [MultipartPart] = []
        for i in 0..<starts.count - 1 {
            var from = starts[i] + delimiter.count
            let to = starts[i + 1]
            // Each part starts with the CRLF after the delimiter and ends with the CRLF before the next.
            if from + 2 <= to, bytes[from] == 0x0D, bytes[from + 1] == 0x0A { from += 2 }
            guard from < to,
                let headerEnd = search(separator, in: bytes, from: from, to: to).first
            else { continue }

            let headers = String(decoding: bytes[from..<headerEnd], as: UTF8.self)
            guard let name = value(of: "name", in: headers) else { continue }

            var bodyEnd = to
            if bodyEnd - 2 >= from, bytes[bodyEnd - 2] == 0x0D, bytes[bodyEnd - 1] == 0x0A {
                bodyEnd -= 2
            }
            let bodyStart = headerEnd + separator.count
            guard bodyStart <= bodyEnd else { continue }

            parts.append(
                MultipartPart(
                    name: name, filename: value(of: "filename", in: headers),
                    body: Array(bytes[bodyStart..<bodyEnd])))
        }
        return parts
    }

    /// Body scan runs over the whole upload, so compare in place: slicing out an Array at
    /// every candidate position turned a 26 MB request into a second of CPU.
    private static func search(_ needle: [UInt8], in hay: [UInt8], from: Int, to: Int) -> [Int] {
        let last = to - needle.count
        guard last >= from, !needle.isEmpty else { return [] }

        var found: [Int] = []
        hay.withUnsafeBufferPointer { buffer in
            needle.withUnsafeBufferPointer { pattern in
                let first = pattern[0]
                var i = from
                while i <= last {
                    // Оба массива непустые, внутри withUnsafeBufferPointer адрес есть.
                    // swiftlint:disable force_unwrapping
                    if buffer[i] == first,
                        memcmp(buffer.baseAddress! + i, pattern.baseAddress!, pattern.count) == 0
                    {
                        // swiftlint:enable force_unwrapping
                        found.append(i)
                        i += pattern.count
                    } else {
                        i += 1
                    }
                }
            }
        }
        return found
    }

    private static func value(of key: String, in headers: String) -> String? {
        guard let range = headers.range(of: "\(key)=\"") else { return nil }
        let rest = headers[range.upperBound...]
        guard let close = rest.firstIndex(of: "\"") else { return nil }
        return String(rest[..<close])
    }
}
