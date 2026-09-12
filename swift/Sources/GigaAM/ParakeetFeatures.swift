// Parakeet feature extraction and attention primitives.
// Log-mel matches the NeMo preprocessing of parakeet_mlx (preemph, reflect-padded
// STFT, slaney filterbanks loaded from the conversion script); the attention carries
// relative positional bias instead of RoPE.

import Foundation
import MLX
import MLXNN

// MARK: - Log-mel (NeMo preprocessing: reflect-padded STFT, |re|+|im| magnitude)

class ParakeetMel {
    static let sampleRate = 16000
    static let nFFT = 512
    static let hopLength = 160
    static let winLength = 400
    static let preemph: Float = 0.97
    static let nMels = 128
    static let reflectPad = nFFT / 2

    let window: MLXArray  // (winLength,)
    let filterbanks: MLXArray  // (nMels, nFFT/2+1)

    /// filterbanks come from librosa (slaney norm) via the conversion script - the
    /// formula is easy to get subtly wrong, so Swift loads the exact array.
    init(filterbanks: MLXArray) {
        self.filterbanks = filterbanks
        // np.hanning(winLength + 1)[:-1] == periodic Hann of length winLength.
        let n = Self.winLength
        let hann = MLXArray((0..<n).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(n)) })
        // win_length < n_fft: the Python STFT zero-pads the window to n_fft.
        window = padded(hann, widths: [IntOrPair((0, Self.nFFT - Self.winLength))])
    }

    /// Preemphasis and reflect padding done on the CPU over raw samples: the padded
    /// array is [x[1..p] reversed, x, x[n-p-1..n-2] reversed] over preemphasized x.
    func preprocess(_ audio: [Float]) -> [Float] {
        let n = audio.count
        guard n > 0 else { return [] }
        let p = Self.reflectPad
        var x = [Float](repeating: 0, count: n)
        x[0] = audio[0]
        for i in 1..<n {
            x[i] = audio[i] - Self.preemph * audio[i - 1]
        }
        var out = [Float](repeating: 0, count: n + 2 * p)
        for j in 0..<p {
            out[p - 1 - j] = x[1 + j]
            out[p + n + j] = x[n - 2 - j]
        }
        out.replaceSubrange(p..<(p + n), with: x)
        return out
    }

    /// padded samples from `preprocess`. Returns (1, T, 128) log-mel, per-feature normalized.
    func callAsFunction(_ padded: [Float]) -> MLXArray {
        let x = MLXArray(padded)
        let n = padded.count
        let nFrames = (n - Self.winLength + Self.hopLength) / Self.hopLength
        let frames = MLX.asStrided(
            x, [nFrames, Self.nFFT], strides: [Self.hopLength, 1], offset: 0)
        let spectrum = MLXFFT.rfft(frames * window, axis: -1)

        // The Python port reads the magnitude as abs of the float view of the
        // complex buffer: |re| + |im| per bin, then squares it. Replicated here
        // bit for bit - a proper sqrt(re^2+im^2) would change the features.
        let floats = spectrum.view(dtype: .float32).reshaped(nFrames, Self.nFFT / 2 + 1, 2)
        let mag = MLX.abs(floats[0..., 0..., 0]) + MLX.abs(floats[0..., 0..., 1])
        let power = mag.square()

        var mel = MLX.matmul(filterbanks, power.transposed())  // (nMels, T)
        mel = MLX.log(mel + 1e-5)

        // per_feature normalization: mean/std over time for every mel band.
        let mean = MLX.mean(mel, axis: 1, keepDims: true)
        let std = MLX.sqrt(MLX.mean(MLX.square(mel - mean), axis: 1, keepDims: true))
        mel = (mel - mean) / (std + 1e-5)

        return mel.transposed().expandedDimensions(axis: 0)  // (1, T, nMels)
    }
}

// MARK: - Relative positional encoding (full attention)

/// Not a Module: the PE table is a constant, not a checkpoint parameter - keeping it
/// inside a Module would make verify(.all) demand an "encoder.posEnc.pe" weight.
final class ParakeetRelPositionalEncoding {
    let dModel: Int
    let maxLen: Int
    /// (1, 2*maxLen-1, dModel) sin/cos table, grown on demand like the Python port.
    var pe: MLXArray

    init(dModel: Int, maxLen: Int) {
        self.dModel = dModel
        self.maxLen = maxLen
        pe = Self.calculate(dModel: dModel, maxLen: maxLen)
    }

    private static func calculate(dModel: Int, maxLen: Int) -> MLXArray {
        let positions = MLXArray(
            stride(from: maxLen - 1, through: -(maxLen - 1), by: -1).map { Float($0) }
        ).reshaped(-1, 1)
        let divTerm = MLX.exp(
            MLXArray(stride(from: 0, to: dModel, by: 2).map { Float($0) })
                * -Float(log(10000.0) / Double(dModel)))
        let angles = positions * divTerm  // (2*maxLen-1, dModel/2)
        // pe[:, 0::2] = sin, pe[:, 1::2] = cos - interleave via a trailing axis.
        let sin = MLX.sin(angles).expandedDimensions(axis: 2)
        let cos = MLX.cos(angles).expandedDimensions(axis: 2)
        return MLX.concatenated([sin, cos], axis: 2).reshaped(-1, dModel).expandedDimensions(
            axis: 0)
    }

    /// Returns (x, pos_emb) where pos_emb covers positions -(inputLen-1)...(inputLen-1).
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let inputLen = x.dim(1)
        if inputLen > maxLen {
            pe = Self.calculate(dModel: dModel, maxLen: inputLen + 1)
        }
        let bufferLen = pe.dim(1)
        let start = bufferLen / 2 - (inputLen - 1)
        let end = bufferLen / 2 + (inputLen - 1) + 1
        return (x, pe[0..., start..<end])
    }
}

// MARK: - Multi-head attention with relative positions

class ParakeetAttention: Module {
    let nHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "linear_pos") var linearPos: Linear
    @ParameterInfo(key: "pos_bias_u") var posBiasU: MLXArray
    @ParameterInfo(key: "pos_bias_v") var posBiasV: MLXArray

    init(nHeads: Int, nFeat: Int) {
        self.nHeads = nHeads
        self.headDim = nFeat / nHeads
        self.scale = pow(Float(headDim), -0.5)
        self._linearQ.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._linearK.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._linearV.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._linearPos.wrappedValue = Linear(nFeat, nFeat, bias: false)
        self._posBiasU.wrappedValue = MLXArray.zeros([nHeads, headDim])
        self._posBiasV.wrappedValue = MLXArray.zeros([nHeads, headDim])
    }

    /// Classic XLNet rel-shift: move position scores so axis -1 aligns query with key.
    private func relShift(_ x: MLXArray) -> MLXArray {  // (B, H, Tq, posLen)
        let (b, h, tq, posLen) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var y = padded(
            x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((1, 0))])
        y = y.reshaped(b, h, posLen + 1, tq)
        y = y[0..., 0..., 1..., 0...]
        return y.reshaped(b, h, tq, posLen)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let qSeq = x.dim(1)

        let q = linearQ(x)
        let k = linearK(x)
        let v = linearV(x)
        let p = linearPos(posEmb)

        let qRaw = q.reshaped(b, qSeq, nHeads, headDim)
        let qU = (qRaw + posBiasU).transposed(0, 2, 1, 3)  // content addressing
        let qV = (qRaw + posBiasV).transposed(0, 2, 1, 3)  // position addressing
        let kH = k.reshaped(b, qSeq, nHeads, headDim).transposed(0, 2, 1, 3)
        let vH = v.reshaped(b, qSeq, nHeads, headDim).transposed(0, 2, 1, 3)
        let pH = p.reshaped(b, -1, nHeads, headDim).transposed(0, 2, 1, 3)

        var matrixBD = MLX.matmul(qV, pH.transposed(0, 1, 3, 2))
        matrixBD = relShift(matrixBD)[0..., 0..., 0..., 0..<qSeq]

        let scores = MLX.matmul(qU, kH.transposed(0, 1, 3, 2)) * scale + matrixBD * scale
        let attn = MLX.softmax(scores, axis: -1)
        let out = MLX.matmul(attn, vH)  // (B, H, Tq, D)

        return linearOut(out.transposed(0, 2, 1, 3).reshaped(b, qSeq, nHeads * headDim))
    }
}
