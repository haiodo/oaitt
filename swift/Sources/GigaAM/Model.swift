// GigaAM v3 e2e — Conformer encoder + CTC/RNNT head on MLX Swift.
// Port of vendor/gigaam-mlx/gigaam_mlx/model.py — weight keys must stay identical.

import Foundation
import MLX
import MLXNN

// MARK: - Rotary positional encoding

func createRotaryPE(length: Int, dim: Int, base: Float = 5000) -> (MLXArray, MLXArray) {
    let idx = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) })
    let invFreq = MLX.exp(-Float(log(Double(base))) * (idx / Float(dim)))
    let t = MLXArray((0..<length).map { Float($0) })
    let freqs = t.reshaped(-1, 1) * invFreq.reshaped(1, -1)
    let emb = MLX.concatenated([freqs, freqs], axis: -1)
    return (MLX.cos(emb), MLX.sin(emb))
}

private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let d = x.dim(-1) / 2
    return MLX.concatenated([-x[.ellipsis, d...], x[.ellipsis, ..<d]], axis: -1)
}

/// Applies RoPE to q, k of shape (T, B, H, D).
private func applyRotary(
    _ q: MLXArray, _ k: MLXArray, _ cos: MLXArray, _ sin: MLXArray
) -> (MLXArray, MLXArray) {
    let t = q.dim(0)
    let c = cos[..<t, .newAxis, .newAxis, 0...]
    let s = sin[..<t, .newAxis, .newAxis, 0...]
    return (q * c + rotateHalf(q) * s, k * c + rotateHalf(k) * s)
}

// MARK: - Conformer blocks

class ConformerFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dModel: Int, dFF: Int) {
        self._linear1.wrappedValue = Linear(dModel, dFF)
        self._linear2.wrappedValue = Linear(dFF, dModel)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

class ConformerConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: LayerNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(dModel: Int, kernelSize: Int) {
        let padding = (kernelSize - 1) / 2
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel * 2, kernelSize: 1)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: kernelSize,
            padding: padding, groups: dModel)
        self._batchNorm.wrappedValue = LayerNorm(dimensions: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(pointwiseConv1(x), parts: 2, axis: -1)
        var y = parts[0] * MLX.sigmoid(parts[1])  // GLU
        y = depthwiseConv(y)
        y = batchNorm(y)
        y = silu(y)
        return pointwiseConv2(y)
    }
}

class RotaryMultiHeadAttention: Module {
    let h: Int
    let dK: Int
    @ModuleInfo(key: "linear_q") var linearQ: Linear
    @ModuleInfo(key: "linear_k") var linearK: Linear
    @ModuleInfo(key: "linear_v") var linearV: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(nHead: Int, nFeat: Int) {
        self.h = nHead
        self.dK = nFeat / nHead
        self._linearQ.wrappedValue = Linear(nFeat, nFeat)
        self._linearK.wrappedValue = Linear(nFeat, nFeat)
        self._linearV.wrappedValue = Linear(nFeat, nFeat)
        self._linearOut.wrappedValue = Linear(nFeat, nFeat)
    }

    func callAsFunction(
        query: MLXArray, key: MLXArray, value: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let (b, t, d) = (query.dim(0), query.dim(1), query.dim(2))

        // RoPE is applied to the raw input, before the linear projections.
        var qRaw = query.reshaped(b, t, h, dK).transposed(1, 0, 2, 3)
        var kRaw = key.reshaped(b, t, h, dK).transposed(1, 0, 2, 3)
        let vRaw = value.reshaped(b, t, h, dK).transposed(1, 0, 2, 3)
        (qRaw, kRaw) = applyRotary(qRaw, kRaw, cos, sin)

        let qIn = qRaw.transposed(1, 0, 2, 3).reshaped(b, t, d)
        let kIn = kRaw.transposed(1, 0, 2, 3).reshaped(b, t, d)
        let vIn = vRaw.transposed(1, 0, 2, 3).reshaped(b, t, d)

        let q = linearQ(qIn).reshaped(b, t, h, dK).transposed(0, 2, 1, 3)
        let k = linearK(kIn).reshaped(b, t, h, dK).transposed(0, 2, 1, 3)
        let v = linearV(vIn).reshaped(b, t, h, dK).transposed(0, 2, 1, 3)

        let scores = matmul(q, k.transposed(0, 1, 3, 2)) / sqrt(Float(dK))
        let out = matmul(MLX.softmax(scores, axis: -1), v)
        return linearOut(out.transposed(0, 2, 1, 3).reshaped(b, t, h * dK))
    }
}

class ConformerLayer: Module {
    let fcFactor: Float = 0.5
    @ModuleInfo(key: "norm_feed_forward1") var normFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ConformerConvolution
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: RotaryMultiHeadAttention
    @ModuleInfo(key: "norm_feed_forward2") var normFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(dModel: Int, dFF: Int, nHeads: Int, convKernelSize: Int) {
        self._normFF1.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward1.wrappedValue = ConformerFeedForward(dModel: dModel, dFF: dFF)
        self._normConv.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = ConformerConvolution(dModel: dModel, kernelSize: convKernelSize)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = RotaryMultiHeadAttention(nHead: nHeads, nFeat: dModel)
        self._normFF2.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward2.wrappedValue = ConformerFeedForward(dModel: dModel, dFF: dFF)
        self._normOut.wrappedValue = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var residual = x + feedForward1(normFF1(x)) * fcFactor

        let normed = normSelfAtt(residual)
        residual += selfAttn(query: normed, key: normed, value: normed, cos: cos, sin: sin)
        residual += conv(normConv(residual))
        residual += feedForward2(normFF2(residual)) * fcFactor

        return normOut(residual)
    }
}

/// 2x Conv1d with stride 2 each -> 4x subsampling.
class Conv1dSubsampling: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv1d
    @ModuleInfo(key: "conv2") var conv2: Conv1d

    init(featIn: Int, featOut: Int, kernelSize: Int = 5) {
        let padding = (kernelSize - 1) / 2
        self._conv1.wrappedValue = Conv1d(
            inputChannels: featIn, outputChannels: featOut, kernelSize: kernelSize,
            stride: 2, padding: padding)
        self._conv2.wrappedValue = Conv1d(
            inputChannels: featOut, outputChannels: featOut, kernelSize: kernelSize,
            stride: 2, padding: padding)
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, Int) {
        let y = relu(conv2(relu(conv1(x))))
        return (y, y.dim(1))
    }
}

class ConformerEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: Conv1dSubsampling
    @ModuleInfo(key: "layers") var layers: [ConformerLayer]
    let ropeDim: Int

    init(
        featIn: Int = 64, nLayers: Int = 16, dModel: Int = 768, nHeads: Int = 16,
        ffExpansionFactor: Int = 4, convKernelSize: Int = 5, subsKernelSize: Int = 5
    ) {
        self._preEncode.wrappedValue = Conv1dSubsampling(
            featIn: featIn, featOut: dModel, kernelSize: subsKernelSize)
        self._layers.wrappedValue = (0..<nLayers).map { _ in
            ConformerLayer(
                dModel: dModel, dFF: dModel * ffExpansionFactor, nHeads: nHeads,
                convKernelSize: convKernelSize)
        }
        self.ropeDim = dModel / nHeads
    }

    func callAsFunction(_ features: MLXArray) -> (MLXArray, Int) {
        var (x, seqLen) = preEncode(features)
        let (cos, sin) = createRotaryPE(length: seqLen, dim: ropeDim)
        for layer in layers {
            x = layer(x, cos: cos, sin: sin)
        }
        return (x.transposed(0, 2, 1), seqLen)
    }
}

// MARK: - Heads

class CTCHead: Module {
    @ModuleInfo(key: "decoder_layers") var decoderLayers: Conv1d

    init(featIn: Int = 768, numClasses: Int = 257) {
        self._decoderLayers.wrappedValue = Conv1d(
            inputChannels: featIn, outputChannels: numClasses, kernelSize: 1)
    }

    func callAsFunction(_ encoderOutput: MLXArray) -> MLXArray {
        let logits = decoderLayers(encoderOutput.transposed(0, 2, 1))
        return logits - MLX.logSumExp(logits, axis: -1, keepDims: true)
    }
}

class RNNTDecoder: Module {
    let predHidden: Int
    let blankId: Int
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "lstm") var lstm: LSTM

    init(predHidden: Int = 320, numClasses: Int = 1025) {
        self.predHidden = predHidden
        self.blankId = numClasses - 1
        self._embed.wrappedValue = Embedding(embeddingCount: numClasses, dimensions: predHidden)
        self._lstm.wrappedValue = LSTM(inputSize: predHidden, hiddenSize: predHidden)
    }

    /// Batched prediction step. `labels` is (B, 1) int32, `hasLabel` is (B, 1, 1) 0/1 -
    /// rows without a previous label feed a zero embedding, matching `predict(nil, ...)`.
    /// A zero hidden/cell state is equivalent to passing none, so the state is never optional here.
    func predictBatch(
        labels: MLXArray, hasLabel: MLXArray, hidden: MLXArray, cell: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray) {
        let emb = embed(labels) * hasLabel
        let (allHidden, allCell) = lstm(emb, hidden: hidden, cell: cell)
        return (allHidden, allHidden[0..., -1, 0...], allCell[0..., -1, 0...])
    }

    func predict(
        _ x: MLXArray?, state: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let emb = x.map { embed($0) } ?? MLXArray.zeros([1, 1, predHidden])
        let (allHidden, allCell) =
            state.map { lstm(emb, hidden: $0.0, cell: $0.1) } ?? lstm(emb)
        return (allHidden, (allHidden[0..., -1, 0...], allCell[0..., -1, 0...]))
    }
}

class RNNTJoint: Module {
    @ModuleInfo(key: "enc_proj") var encProj: Linear
    @ModuleInfo(key: "pred_proj") var predProj: Linear
    @ModuleInfo(key: "out") var out: Linear

    init(
        encHidden: Int = 768, predHidden: Int = 320, jointHidden: Int = 320, numClasses: Int = 1025
    ) {
        self._encProj.wrappedValue = Linear(encHidden, jointHidden)
        self._predProj.wrappedValue = Linear(predHidden, jointHidden)
        self._out.wrappedValue = Linear(jointHidden, numClasses)
    }

    func callAsFunction(_ enc: MLXArray, _ pred: MLXArray) -> MLXArray {
        let e = encProj(enc).expandedDimensions(axis: 2)
        let p = predProj(pred).expandedDimensions(axis: 1)
        let logits = out(relu(e + p))
        return logits - MLX.logSumExp(logits, axis: -1, keepDims: true)
    }
}

// MARK: - Full model

public enum GigaAMModelType: String, Sendable, CaseIterable {
    case ctc, rnnt

    var numClasses: Int { self == .ctc ? 257 : 1025 }
}

public class GigaAMMLX: Module {
    public let modelType: GigaAMModelType
    @ModuleInfo(key: "encoder") var encoder: ConformerEncoder
    @ModuleInfo(key: "head") var head: CTCHead?
    @ModuleInfo(key: "decoder") var decoder: RNNTDecoder?
    @ModuleInfo(key: "joint") var joint: RNNTJoint?

    public init(modelType: GigaAMModelType = .ctc) {
        self.modelType = modelType
        self._encoder.wrappedValue = ConformerEncoder()
        switch modelType {
        case .ctc:
            self._head.wrappedValue = CTCHead(numClasses: 257)
        case .rnnt:
            self._decoder.wrappedValue = RNNTDecoder(predHidden: 320, numClasses: 1025)
            self._joint.wrappedValue = RNNTJoint()
        }
    }

    /// Runs the conformer encoder. Input: (B, T, 64) log-mel.
    public func encode(_ features: MLXArray) -> (MLXArray, Int) {
        encoder(features)
    }

    public func decode(_ encoded: MLXArray, seqLen: Int) -> [Int] {
        modelType == .ctc ? ctcDecode(encoded, seqLen) : rnntDecode(encoded, seqLen)
    }

    func ctcDecode(_ encoded: MLXArray, _ seqLen: Int) -> [Int] {
        guard let head else { return [] }
        let logProbs = head(encoded)
        let labels = MLX.argMax(logProbs[0, ..<seqLen, 0...], axis: -1)
        let blankId = modelType.numClasses - 1

        var tokens: [Int] = []
        var prev = blankId
        for tok in labels.asArray(Int32.self).map(Int.init) {
            if tok != blankId && tok != prev { tokens.append(tok) }
            prev = tok
        }
        return tokens
    }

    func rnntDecode(_ encoded: MLXArray, _ seqLen: Int, maxSymbols: Int = 10) -> [Int] {
        guard let decoder, let joint else { return [] }
        let enc = encoded[0]  // (C, T)
        let blankId = decoder.blankId
        var hyp: [Int] = []
        var state: (MLXArray, MLXArray)?
        var lastLabel: MLXArray?

        for t in 0..<seqLen {
            let f = enc[0..., t..<(t + 1)].transposed().expandedDimensions(axis: 0)
            var symbols = 0
            while symbols < maxSymbols {
                let (g, newState) = decoder.predict(lastLabel, state: state)
                let logits = joint(f, g)
                let k = Int(MLX.argMax(logits[0, 0, 0, 0...]).item(Int32.self))
                if k == blankId { break }
                hyp.append(k)
                state = newState
                lastLabel = MLXArray([Int32(k)]).reshaped(1, 1)
                symbols += 1
            }
        }
        return hyp
    }

    /// Greedy RNNT decode over several encoder outputs at once.
    ///
    /// Per-chunk decoding needs one GPU->CPU sync per frame to read the argmax; batching the
    /// chunks amortises those syncs across B hypotheses. `encodedList` entries are (1, C, T).
    public func rnntDecodeBatch(
        _ encodedList: [MLXArray], seqLens: [Int], maxSymbols: Int = 10
    ) -> [[Int]] {
        guard let decoder, let joint, !encodedList.isEmpty else { return [] }
        let batch = encodedList.count
        let hiddenSize = decoder.predHidden
        let blankId = decoder.blankId
        let maxT = seqLens.max() ?? 0
        let channels = encodedList[0].dim(1)

        let padded = MLX.concatenated(
            encodedList.map { enc in
                let missing = maxT - enc.dim(2)
                return missing > 0
                    ? MLX.concatenated([enc, MLXArray.zeros([1, channels, missing])], axis: 2)
                    : enc[0..., 0..., ..<maxT]
            }, axis: 0)  // (B, C, maxT)

        var hypotheses = [[Int]](repeating: [], count: batch)
        var labelsCPU = [Int32](repeating: 0, count: batch)
        var hasLabelCPU = [Float](repeating: 0, count: batch)
        var hidden = MLXArray.zeros([batch, hiddenSize])
        var cell = MLXArray.zeros([batch, hiddenSize])

        for t in 0..<maxT {
            let f = padded[0..., 0..., t].expandedDimensions(axis: 1)  // (B, 1, C)
            var notBlank = (0..<batch).map { t < seqLens[$0] }
            var symbols = 0

            while symbols < maxSymbols && notBlank.contains(true) {
                let labels = MLXArray(labelsCPU).reshaped(batch, 1)
                let hasLabel = MLXArray(hasLabelCPU).reshaped(batch, 1, 1)
                let (g, newHidden, newCell) = decoder.predictBatch(
                    labels: labels, hasLabel: hasLabel, hidden: hidden, cell: cell)
                let best = MLX.argMax(joint(f, g)[0..., 0, 0, 0...], axis: -1).asArray(Int32.self)

                var advanced = [Float](repeating: 0, count: batch)
                for b in 0..<batch where notBlank[b] {
                    if best[b] == Int32(blankId) {
                        notBlank[b] = false
                    } else {
                        hypotheses[b].append(Int(best[b]))
                        labelsCPU[b] = best[b]
                        hasLabelCPU[b] = 1
                        advanced[b] = 1
                    }
                }
                symbols += 1
                guard advanced.contains(1) else { break }

                // Only rows that emitted a symbol advance their LSTM state.
                let mask = MLXArray(advanced).reshaped(batch, 1) .> 0
                hidden = MLX.which(mask, newHidden, hidden)
                cell = MLX.which(mask, newCell, cell)
            }
        }
        return hypotheses
    }
}
