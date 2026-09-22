// NVIDIA Parakeet-TDT-v3 (600M, 25 European languages) on MLX Swift.
// Port of the parakeet_mlx Python package - weight keys must stay identical to
// data/parakeet_tdt_v3/weights.safetensors (converted by
// scripts/convert_parakeet_to_mlx_swift.py from mlx-community/parakeet-tdt-0.6b-v3).
//
// Architecture: FastConformer encoder (24 layers, d_model 1024, 8 heads, rel_pos
// attention, depthwise-conv subsampling x8) + TDT predictor/joint with a duration
// head. Unlike GigaAM there is no RoPE - relative positional embeddings enter the
// attention as an additive bias, and the decoder can skip frames (durations 0-4).

import Foundation
import MLX
import MLXNN

private let parakeetDebug = ProcessInfo.processInfo.environment["PARAKEET_DEBUG"] != nil

// MARK: - Conformer block

class ParakeetFeedForward: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(dModel: Int, dFF: Int) {
        self._linear1.wrappedValue = Linear(dModel, dFF, bias: false)
        self._linear2.wrappedValue = Linear(dFF, dModel, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(x)))
    }
}

class ParakeetConvolution: Module {
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

    init(dModel: Int, kernelSize: Int) {
        let padding = (kernelSize - 1) / 2
        // use_bias=false in the checkpoint: the convs carry no bias terms.
        self._pointwiseConv1.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel * 2, kernelSize: 1, bias: false)
        self._depthwiseConv.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: kernelSize,
            groups: dModel, bias: false)
        self._batchNorm.wrappedValue = BatchNorm(featureCount: dModel)
        self._pointwiseConv2.wrappedValue = Conv1d(
            inputChannels: dModel, outputChannels: dModel, kernelSize: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {  // (B, T, C) layout
        let parts = MLX.split(pointwiseConv1(x), parts: 2, axis: -1)
        var y = parts[0] * MLX.sigmoid(parts[1])  // GLU
        y = padded(y, widths: [IntOrPair((0, 0)), IntOrPair((4, 4)), IntOrPair((0, 0))])
        y = depthwiseConv(y)
        y = batchNorm(y)
        y = silu(y)
        return pointwiseConv2(y)
    }
}

class ParakeetConformerBlock: Module {
    let ffFactor: Float = 0.5

    @ModuleInfo(key: "norm_feed_forward1") var normFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var feedForward1: ParakeetFeedForward
    @ModuleInfo(key: "norm_self_att") var normSelfAtt: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttn: ParakeetAttention
    @ModuleInfo(key: "norm_conv") var normConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ParakeetConvolution
    @ModuleInfo(key: "norm_feed_forward2") var normFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var feedForward2: ParakeetFeedForward
    @ModuleInfo(key: "norm_out") var normOut: LayerNorm

    init(dModel: Int, dFF: Int, nHeads: Int, convKernelSize: Int) {
        self._normFF1.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward1.wrappedValue = ParakeetFeedForward(dModel: dModel, dFF: dFF)
        self._normSelfAtt.wrappedValue = LayerNorm(dimensions: dModel)
        self._selfAttn.wrappedValue = ParakeetAttention(nHeads: nHeads, nFeat: dModel)
        self._normConv.wrappedValue = LayerNorm(dimensions: dModel)
        self._conv.wrappedValue = ParakeetConvolution(dModel: dModel, kernelSize: convKernelSize)
        self._normFF2.wrappedValue = LayerNorm(dimensions: dModel)
        self._feedForward2.wrappedValue = ParakeetFeedForward(dModel: dModel, dFF: dFF)
        self._normOut.wrappedValue = LayerNorm(dimensions: dModel)
    }

    func callAsFunction(_ x: MLXArray, posEmb: MLXArray) -> MLXArray {
        var residual = x + feedForward1(normFF1(x)) * ffFactor
        let normed = normSelfAtt(residual)
        residual += selfAttn(normed, posEmb: posEmb)
        residual += conv(normConv(residual))
        residual += feedForward2(normFF2(residual)) * ffFactor
        return normOut(residual)
    }
}

// MARK: - Depthwise striding subsampling (x8)

class ParakeetSubsampling: Module {
    /// The Python pre_encode.conv is a Sequential LIST: Conv3x3, ReLU, DwConv3x3,
    /// Conv1x1, ReLU, DwConv3x3, Conv1x1, ReLU. Numeric keys unfold into arrays on the
    /// weights side, so the model tree mirrors that - with nil at the ReLU slots.
    @ModuleInfo(key: "conv") var conv: [Module?]
    @ModuleInfo(key: "out") var out: Linear

    /// Typed view over the weighted slots; NOT a stored property - Module reflects
    /// every stored child into the parameter tree.
    private var convs: [Conv2d] { conv.compactMap { $0 as? Conv2d } }

    init(featIn: Int, channels: Int, dModel: Int) {
        let c0 = Conv2d(
            inputChannels: 1, outputChannels: channels, kernelSize: 3, stride: 2, padding: 1)
        let c2 = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2,
            padding: 1, groups: channels)
        let c3 = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        let c5 = Conv2d(
            inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2,
            padding: 1, groups: channels)
        let c6 = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        self._conv.wrappedValue = [c0, nil, c2, c3, nil, c5, c6, nil]
        self._out.wrappedValue = Linear(channels * Self.outputFreq(featIn: featIn), dModel)
    }

    static func outputFreq(featIn: Int) -> Int {
        var f = featIn
        for _ in 0..<3 {
            f = (f + 2 - 3) / 2 + 1
        }
        return f
    }

    func callAsFunction(_ x: MLXArray, lengths: MLXArray) -> (MLXArray, MLXArray) {
        // x: (B, T, featIn) -> (B, T, F, 1) NHWC as the Python port lays it out.
        var y = x.expandedDimensions(axis: 3)  // (B, H=T, W=F, C=1)
        // Sequential: Conv, ReLU, DwConv, Conv1x1, ReLU, DwConv, Conv1x1, ReLU.
        y = relu(convs[0](y))
        y = convs[2](convs[1](y))
        y = relu(y)
        y = convs[4](convs[3](y))
        y = relu(y)
        // (B, H=T, W=F, C) -> (B, T, C, F) -> (B, T, C*F), channel-major like upstream
        y = y.transposed(0, 1, 3, 2).reshaped(y.dim(0), y.dim(1), -1)
        y = out(y)

        var len = lengths
        for _ in 0..<3 {
            len = MLX.floor((len + 2 - 3) / 2) + 1
        }
        return (y, len.asType(.int32))
    }
}

// MARK: - Encoder

class ParakeetEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: ParakeetSubsampling
    @ModuleInfo(key: "layers") var layers: [ParakeetConformerBlock]
    let posEnc: ParakeetRelPositionalEncoding

    override init() {
        self._preEncode.wrappedValue = ParakeetSubsampling(featIn: 128, channels: 256, dModel: 1024)
        self._layers.wrappedValue = (0..<24).map { _ in
            ParakeetConformerBlock(dModel: 1024, dFF: 4096, nHeads: 8, convKernelSize: 9)
        }
        self.posEnc = ParakeetRelPositionalEncoding(dModel: 1024, maxLen: 5000)
    }

    /// Input: (1, T, 128) log-mel. Returns (1, T', 1024) and the output length.
    func callAsFunction(_ mel: MLXArray) -> (MLXArray, Int) {
        let lengths = MLXArray([Float(mel.dim(1))])
        let (x, outLengths) = preEncode(mel, lengths: lengths)
        let (_, posEmb) = posEnc(x)
        var hidden = x
        for layer in layers {
            hidden = layer(hidden, posEmb: posEmb)
        }
        return (hidden, Int(outLengths[0].item(Int32.self)))
    }
}

// MARK: - TDT predictor and joint

class ParakeetPredictionInner: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "dec_rnn") var decRnn: ParakeetDecRNN

    init(predHidden: Int) {
        self._embed.wrappedValue = Embedding(embeddingCount: 8193, dimensions: predHidden)
        self._decRnn.wrappedValue = ParakeetDecRNN(
            inputSize: predHidden, hiddenSize: predHidden)
    }
}

class ParakeetPredictor: Module {
    static let predHidden = 640

    @ModuleInfo(key: "prediction") var prediction: ParakeetPredictionInner

    override init() {
        self._prediction.wrappedValue = ParakeetPredictionInner(predHidden: Self.predHidden)
    }

    /// None input feeds a zero embedding, matching `predict(nil, ...)` upstream.
    func callAsFunction(
        _ lastToken: Int32?, state: ([MLXArray], [MLXArray])?
    ) -> (MLXArray, ([MLXArray], [MLXArray])) {
        let embedded: MLXArray
        if let lastToken {
            embedded = prediction.embed(MLXArray([lastToken]).reshaped(1, 1))
        } else {
            embedded = MLXArray.zeros([1, 1, Self.predHidden])
        }
        let (out, newState) = prediction.decRnn(
            embedded,
            hidden: state?.0,
            cell: state?.1)
        return (out, newState)
    }
}

/// dec_rnn wraps the two LSTM layers under key "lstm" - checkpoint keys are
/// dec_rnn.lstm.<i>.{Wx,Wh,bias}. Per-layer states are the last step of each
/// layer's output, like the Python port.
class ParakeetDecRNN: Module {
    /// dec_rnn.lstm is a two-element list in the checkpoint: lstm.0 / lstm.1.
    @ModuleInfo(key: "lstm") var lstm: [LSTM]

    init(inputSize: Int, hiddenSize: Int) {
        self._lstm.wrappedValue = [
            LSTM(inputSize: inputSize, hiddenSize: hiddenSize),
            LSTM(inputSize: hiddenSize, hiddenSize: hiddenSize),
        ]
    }

    func callAsFunction(
        _ x: MLXArray, hidden: [MLXArray]?, cell: [MLXArray]?
    ) -> (MLXArray, ([MLXArray], [MLXArray])) {
        var out = x
        var nextH: [MLXArray] = []
        var nextC: [MLXArray] = []
        for (i, layer) in lstm.enumerated() {
            let (allHidden, allCell) = layer(out, hidden: hidden?[i], cell: cell?[i])
            out = allHidden  // whole sequence feeds the next layer
            nextH.append(allHidden[0..., -1, 0...])
            nextC.append(allCell[0..., -1, 0...])
        }
        return (out, (nextH, nextC))
    }
}

class ParakeetJoint: Module {
    @ModuleInfo(key: "enc") var enc: Linear
    @ModuleInfo(key: "pred") var pred: Linear
    /// joint_net is Sequential(relu, Identity, Linear) upstream - a list whose only
    /// weighted slot is index 2.
    @ModuleInfo(key: "joint_net") var jointNet: [Module?]

    private var jointLinear: Linear { jointNet.compactMap { $0 as? Linear }[0] }

    override init() {
        self._enc.wrappedValue = Linear(1024, 640)
        self._pred.wrappedValue = Linear(640, 640)
        self._jointNet.wrappedValue = [nil, nil, Linear(640, 8198)]
    }

    /// e: enc-projected frame (1, 1, 640), p: pred-projected decoder output (1, 1, 640).
    /// Projections are applied by the caller so they are not recomputed every step.
    /// Returns (1, 1, 1, 8198) logits.
    func callAsFunction(_ e: MLXArray, _ p: MLXArray) -> MLXArray {
        let x = e.expandedDimensions(axis: 2) + p.expandedDimensions(axis: 1)
        return jointLinear(relu(x))
    }
}

// MARK: - Full model

struct ParakeetToken {
    let id: Int
    let text: String
    let start: Double
    let duration: Double
    let confidence: Float
    var end: Double { start + duration }
}

class ParakeetTDTModel: Module {
    @ModuleInfo(key: "encoder") var encoder: ParakeetEncoder
    @ModuleInfo(key: "decoder") var decoder: ParakeetPredictor
    @ModuleInfo(key: "joint") var joint: ParakeetJoint

    let vocabulary: [String]
    let durations = [0, 1, 2, 3, 4]
    let timeRatio: Double = 0.08  // 8 frames skip * 160 hop / 16000 Hz
    let maxSymbols = 10
    let mel: ParakeetMel

    init(vocabulary: [String], filterbanks: MLXArray) {
        self._encoder.wrappedValue = ParakeetEncoder()
        self._decoder.wrappedValue = ParakeetPredictor()
        self._joint.wrappedValue = ParakeetJoint()
        self.vocabulary = vocabulary
        self.mel = ParakeetMel(filterbanks: filterbanks)
    }

    func decode(_ token: Int) -> String {
        vocabulary[token].replacingOccurrences(of: "▁", with: " ")
    }

    /// audio: raw 16 kHz samples of ONE chunk. Returns aligned tokens for the chunk.
    func transcribeChunk(_ audio: [Float]) -> [ParakeetToken] {
        let padded = mel.preprocess(audio)
        guard padded.count > ParakeetMel.winLength else { return [] }
        let features = mel(padded)
        let (encoded, seqLen) = encoder(features)
        eval(encoded)
        return decodeGreedy(encoded, seqLen: seqLen)
    }

    /// Greedy TDT decode over encoder output (seqLen, 1024), mirroring
    /// ParakeetTDT.decode_greedy in the Python package.
    private func decodeGreedy(_ features: MLXArray, seqLen: Int) -> [ParakeetToken] {
        var hypothesis: [ParakeetToken] = []
        var lastToken: Int32?
        var state: ([MLXArray], [MLXArray])?

        var step = 0
        var newSymbols = 0
        let vocabLimit = vocabulary.count + 1  // blank sits at index vocabulary.count
        let maxEntropy = log(Float(vocabLimit))

        let encProj = joint.enc(features)  // (1, T, 640)
        eval(encProj)
        // Blank keeps the decoder state, so its output is reused until a token is emitted.
        var predProj: MLXArray?
        var newState: ([MLXArray], [MLXArray])?

        while step < seqLen {
            let proj: MLXArray
            if let cached = predProj {
                proj = cached
            } else {
                let (decoderOut, next) = decoder(lastToken, state: state)
                proj = joint.pred(decoderOut)
                predProj = proj
                newState = next
            }
            let logits = joint(encProj[0..., step..<(step + 1), 0...], proj)

            let row = logits[0, 0, 0, 0...]
            let tokenLogits = row[..<vocabLimit]
            let probs = MLX.softmax(tokenLogits, axis: -1)
            let tokenArr = MLX.argMax(tokenLogits, axis: -1)
            let entropyArr = -MLX.sum(probs * MLX.log(probs + 1e-10), axis: -1)
            let decisionArr = MLX.argMax(row[vocabLimit...], axis: -1)
            // The LSTM state rides along so its lazy graph does not grow token to token.
            eval([tokenArr, entropyArr, decisionArr] + (newState.map { $0.0 + $0.1 } ?? []))

            let predToken = Int(tokenArr.item(Int32.self))
            let confidence = 1.0 - entropyArr.item(Float.self) / maxEntropy
            let decision = Int(decisionArr.item(Int32.self))

            // TDT rule: emit unless blank; advance time by the predicted duration.
            if predToken != vocabulary.count {
                hypothesis.append(
                    ParakeetToken(
                        id: predToken,
                        text: decode(predToken),
                        start: Double(step) * timeRatio,
                        duration: Double(durations[decision]) * timeRatio,
                        confidence: confidence))
                lastToken = Int32(predToken)
                state = newState
                predProj = nil
            }

            step += durations[decision]
            newSymbols += 1

            if durations[decision] != 0 {
                newSymbols = 0
            } else if maxSymbols <= newSymbols {
                step += 1
                newSymbols = 0
            }
        }
        return hypothesis
    }
}
