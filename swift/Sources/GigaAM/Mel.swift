// Log-mel spectrogram matching librosa.feature.melspectrogram(htk=True, norm=None,
// center=False, power=2.0) as used by gigaam_mlx.audio.compute_mel.

import Foundation
import MLX
import MLXFFT

public struct MelSpectrogram {
    public static let sampleRate = 16000
    public static let nMels = 64
    public static let nFFT = 320
    public static let hopLength = 160
    public static let winLength = 320

    private let window: MLXArray  // (nFFT,)
    private let filters: MLXArray  // (nFFT/2+1, nMels)

    public init() {
        let n = Self.nFFT
        // Periodic Hann (librosa's fftbins=True), win_length == n_fft so no padding.
        window = MLXArray((0..<n).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(n)) })
        filters = MLXArray(Self.melFilterBank(), [n / 2 + 1, Self.nMels])
    }

    /// audio: 16 kHz mono float32. Returns (T, 64) log-mel.
    public func callAsFunction(_ audio: [Float]) -> MLXArray {
        let nFrames = 1 + (audio.count - Self.nFFT) / Self.hopLength
        precondition(nFrames > 0, "audio shorter than one FFT frame")

        let x = MLXArray(audio)
        let frames = MLX.asStrided(
            x, [nFrames, Self.nFFT], strides: [Self.hopLength, 1], offset: 0)
        let spectrum = MLXFFT.rfft(frames * window, axis: -1)
        let power = MLX.abs(spectrum).square()
        let mel = MLX.matmul(power, filters)
        return MLX.log(MLX.clip(mel, min: 1e-9, max: 1e9))
    }

    // MARK: - librosa.filters.mel(htk: true, norm: nil), flattened row-major (freq, mel)

    private static func hzToMel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
    private static func melToHz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

    private static func melFilterBank() -> [Float] {
        let nFreqs = nFFT / 2 + 1
        let fMax = Double(sampleRate) / 2
        let fftFreqs = (0..<nFreqs).map { Double($0) * fMax / Double(nFreqs - 1) }

        let melMin = hzToMel(0), melMax = hzToMel(fMax)
        let melPoints = (0..<nMels + 2).map {
            melToHz(melMin + (melMax - melMin) * Double($0) / Double(nMels + 1))
        }

        var weights = [Float](repeating: 0, count: nFreqs * nMels)
        for m in 0..<nMels {
            let (lo, ctr, hi) = (melPoints[m], melPoints[m + 1], melPoints[m + 2])
            for (k, f) in fftFreqs.enumerated() {
                let lower = (f - lo) / (ctr - lo)
                let upper = (hi - f) / (hi - ctr)
                weights[k * nMels + m] = Float(max(0, min(lower, upper)))
            }
        }
        return weights
    }
}
