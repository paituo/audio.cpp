#include "engine/models/whisper/runtime.h"

#include "engine/framework/audio/conversion.h"
#include "engine/framework/audio/dsp.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/attention/cross_attention.h"
#include "engine/framework/modules/attention/feed_forward.h"
#include "engine/framework/modules/attention/self_attention.h"
#include "engine/framework/modules/attention/transformer_blocks.h"
#include "engine/framework/modules/linear_module.h"
#include "engine/framework/modules/lookup_modules.h"
#include "engine/framework/modules/norm_modules.h"
#include "engine/framework/modules/speech_encoders/whisper_embedding.h"
#include "engine/framework/modules/weight_binding.h"
#include "engine/framework/runtime/bounded_static_kv_decode.h"
#include "engine/framework/runtime/graph_optimizer.h"
#include "engine/framework/runtime/kv_cache.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace engine::models::whisper {

namespace {
using core::TensorShape;
using core::TensorValue;
namespace binding = modules::binding;
namespace audio = engine::audio;

// ggml_init 会用 ggml_aligned_malloc 按 mem_size 全量提交 arena（no_alloc 不影响 arena 本身），
// 而 whisper 在这些 ctx 里只放 tensor/图节点元数据（<10MB），1GB/256MB 属严重过量预留。
// 调小到 128MB/64MB 仍留 ~百倍余量，省约 2.3GB 无谓的 host commit（见 Private 归因）。
constexpr size_t kWeightContextBytes = 128ull * 1024ull * 1024ull;
constexpr size_t kGraphContextBytes = 64ull * 1024ull * 1024ull;
constexpr int64_t kSampleRate = 16000;
constexpr int64_t kNfft = 400;
constexpr int64_t kHop = 160;
constexpr int64_t kMaxInputSamples = 480000;  // 30s @ 16k

// ---------------------------------------------------------------- helpers

// Whisper 特殊 token 的【绝对 ID 并非固定】：取决于语言数量 N。
//   vocab = 50257(base bpe) + [2: <|endoftext|>,<|startoftranscript|>]
//           + N 语言 + [6: translate,transcribe,startoflm,startofprev,nospeech,notimestamps]
//           + [1501: <|0.00|> 及 1500 时间戳]
//   → vocab = 50257 + 2 + N + 6 + 1501 = 51766 + N  →  N = vocab - 51766
// 各 task/timestamp token 基于 sot=50258 的偏移：
//   translate = sot + N + 1
//   transcribe = sot + N + 2
//   notimestamps = sot + N + 6
//   timestamp_begin = notimestamps + 1
// 实测校验：
//   openai 官方 multilingual（99 语言, vocab 51865）→ translate=50358, notimestamps=50363,
//     timestamp_begin=50364（audio.cpp 原硬编码值，海南鸡 large-v2 正常）；
//   whisper-ja（100 语言, 独有 <|yue|>=50358, vocab 51866）→ translate=50359, transcribe=50360,
//     notimestamps=50364, timestamp_begin=50365，与 HF tokenizer.json 实测逐一对上。
// ★ 若沿用 99 语言的硬编码 ID，对 100 语言模型的 prompt task/timestamp token 会整体偏小 1，
//   起始上下文错乱 → decoder 第一步即输出 eot(50257)（本次 whisper-ja 根因）。
struct WhisperSpecialLayout {
    int64_t n_lang = 0;
    int32_t translate = 0;
    int32_t transcribe = 0;
    int32_t notimestamps = 0;
    int32_t timestamp_begin = 0;
};

WhisperSpecialLayout whisper_special_layout(int64_t vocab_size) {
    const int64_t N = vocab_size - 51766;  // 语言数量
    const int64_t sot = 50258;
    WhisperSpecialLayout L;
    L.n_lang = N;
    L.translate = static_cast<int32_t>(sot + N + 1);
    L.transcribe = static_cast<int32_t>(sot + N + 2);
    L.notimestamps = static_cast<int32_t>(sot + N + 6);
    L.timestamp_begin = static_cast<int32_t>(L.notimestamps + 1);
    return L;
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::vector<float> pack_qkv(const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v) {
    std::vector<float> out;
    out.reserve(q.size() + k.size() + v.size());
    out.insert(out.end(), q.begin(), q.end());
    out.insert(out.end(), k.begin(), k.end());
    out.insert(out.end(), v.begin(), v.end());
    return out;
}

std::vector<float> pack_kv(const std::vector<float> & k, const std::vector<float> & v) {
    std::vector<float> out;
    out.reserve(k.size() + v.size());
    out.insert(out.end(), k.begin(), k.end());
    out.insert(out.end(), v.begin(), v.end());
    return out;
}

std::vector<float> zeros(size_t n) { return std::vector<float>(n, 0.0F); }

// ---------------------------------------------------------------- log-mel（HTK + Kokoro，whisper 对齐）

double hz_to_mel_htk(double f) { return 2595.0 * std::log10(1.0 + f / 700.0); }
double mel_to_hz_htk(double m) { return 700.0 * (std::pow(10.0, m / 2595.0) - 1.0); }

// whisper/torchaudio HTK mel filterbank（n_fft=400, n_mels=80, sr=16000, norm=slaney）。
audio::AudioTensor build_htk_filterbank(int64_t n_mels, int64_t n_fft, int64_t sr) {
    const int64_t freq_bins = n_fft / 2 + 1;
    const double f_min = 0.0;
    const double f_max = static_cast<double>(sr) / 2.0;
    std::vector<double> mel_pts(static_cast<size_t>(n_mels + 2));
    const double m_min = hz_to_mel_htk(f_min);
    const double m_max = hz_to_mel_htk(f_max);
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        mel_pts[static_cast<size_t>(i)] =
            m_min + (m_max - m_min) * static_cast<double>(i) / static_cast<double>(n_mels + 1);
    }
    std::vector<double> hz_pts(static_cast<size_t>(n_mels + 2));
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        hz_pts[static_cast<size_t>(i)] = mel_to_hz_htk(mel_pts[static_cast<size_t>(i)]);
    }
    std::vector<int64_t> fbins(static_cast<size_t>(n_mels + 2));
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        double b = std::floor(static_cast<double>(n_fft + 1) * hz_pts[static_cast<size_t>(i)] / static_cast<double>(sr));
        fbins[static_cast<size_t>(i)] = static_cast<int64_t>(b);
        if (fbins[static_cast<size_t>(i)] < 0) fbins[static_cast<size_t>(i)] = 0;
        if (fbins[static_cast<size_t>(i)] > freq_bins - 1) fbins[static_cast<size_t>(i)] = freq_bins - 1;
    }
    audio::AudioTensor fbank;
    fbank.shape = {n_mels, freq_bins};
    fbank.values.assign(static_cast<size_t>(n_mels * freq_bins), 0.0f);
    for (int64_t m = 0; m < n_mels; ++m) {
        for (int64_t j = fbins[static_cast<size_t>(m)]; j < fbins[static_cast<size_t>(m + 1)]; ++j) {
            const double denom = static_cast<double>(fbins[static_cast<size_t>(m + 1)] - fbins[static_cast<size_t>(m)]);
            if (denom > 0.0) {
                fbank.values[static_cast<size_t>(m * freq_bins + j)] =
                    static_cast<float>(static_cast<double>(j - fbins[static_cast<size_t>(m)]) / denom);
            }
        }
        for (int64_t j = fbins[static_cast<size_t>(m + 1)]; j < fbins[static_cast<size_t>(m + 2)]; ++j) {
            const double denom = static_cast<double>(fbins[static_cast<size_t>(m + 2)] - fbins[static_cast<size_t>(m + 1)]);
            if (denom > 0.0) {
                fbank.values[static_cast<size_t>(m * freq_bins + j)] =
                    static_cast<float>(static_cast<double>(fbins[static_cast<size_t>(m + 2)] - j) / denom);
            }
        }
    }
    for (int64_t m = 0; m < n_mels; ++m) {
        const double enorm = 2.0 / std::max(hz_pts[static_cast<size_t>(m + 2)] - hz_pts[static_cast<size_t>(m)], 1e-12);
        for (int64_t j = 0; j < freq_bins; ++j) {
            fbank.values[static_cast<size_t>(m * freq_bins + j)] *= static_cast<float>(enorm);
        }
    }
    return fbank;
}

// 由 16k 单声道音频算 whisper 对齐 log-mel（HTK filterbank + Kokoro periodic Hann + log10 floor）。
// 输入会尾部补 0 / 截断到 kMaxInputSamples，输出 [n_mels, frames]（frames≈3000）。
std::vector<float> compute_log_mel(
    const std::vector<float> & mono_16k,
    int64_t n_mels) {
    std::vector<float> samples = mono_16k;
    if (samples.size() > static_cast<size_t>(kMaxInputSamples)) {
        samples.resize(static_cast<size_t>(kMaxInputSamples));
    } else if (samples.size() < static_cast<size_t>(kMaxInputSamples)) {
        samples.resize(static_cast<size_t>(kMaxInputSamples), 0.0f);
    }

    // STFT：与 whisper 对齐用 Kokoro（periodic Hann）窗口 + Reflect padding。
    const audio::STFTConfig stft_config{kNfft, kHop, kNfft, true, audio::STFTPadMode::Reflect, audio::STFTFamily::Kokoro};
    const auto & window = audio::get_cached_stft_window(stft_config);
    auto magnitude = audio::STFT().compute_magnitude(
        samples, window, 1, static_cast<int64_t>(samples.size()), stft_config, 0);
    const int64_t freq_bins = magnitude.shape[1];
    const int64_t stft_frames = magnitude.shape[2];
    const int64_t out_frames = stft_frames - 1;

    auto fbank = build_htk_filterbank(n_mels, kNfft, kSampleRate);
    audio::SparseMelFilterbank sparse = audio::MelFilterbank().prepare_sparse(fbank);
    for (size_t m = 0; m < sparse.starts.size(); ++m) {
        // HTK filterbank 在 n_fft=400 时的全零 mel 行（start>end），语义恒输出 floor；patch 成空区间。
        if (sparse.starts[m] > sparse.ends[m]) {
            sparse.starts[m] = 0;
            sparse.ends[m] = 0;
        }
    }
    auto mel = audio::MelFilterbank().compute_custom_sparse_from_magnitude(
        magnitude.values, 1, freq_bins, stft_frames, out_frames, sparse);
    float max_log = -std::numeric_limits<float>::infinity();
    for (float & value : mel.values) {
        value = std::log10(std::max(value, 1.0e-10F));
        max_log = std::max(max_log, value);
    }
    const float floor = max_log - 8.0F;
    for (float & value : mel.values) {
        value = (std::max(value, floor) + 4.0F) / 4.0F;
    }
    return mel.values;
}

// ---------------------------------------------------------------- encoder 权重（HF 布局）

modules::LinearWeights linear_weights(
    core::BackendWeightStore & store,
    const assets::TensorSource & source,
    const std::string & prefix,
    int64_t out,
    int64_t in,
    bool bias) {
    return binding::linear_from_source(
        store, source, prefix, assets::TensorStorageType::Native, out, in, bias);
}

void load_encoder_weights(
    modules::WhisperEmbeddingWeights & weights,
    const modules::WhisperEmbeddingConfig & config,
    core::BackendWeightStore & store,
    const assets::TensorSource & source) {
    const auto st = assets::TensorStorageType::Native;
    weights.conv1 = {
        store.load_tensor(source, "model.encoder.conv1.weight", st, {config.n_audio_state, config.n_mels, 3}),
        store.load_f32_tensor(source, "model.encoder.conv1.bias", {config.n_audio_state}),
    };
    weights.conv2 = {
        store.load_tensor(source, "model.encoder.conv2.weight", st, {config.n_audio_state, config.n_audio_state, 3}),
        store.load_f32_tensor(source, "model.encoder.conv2.bias", {config.n_audio_state}),
    };
    weights.positional_embedding = store.load_f32_tensor(
        source, "model.encoder.embed_positions.weight", {config.n_audio_ctx, config.n_audio_state});
    weights.layers.reserve(static_cast<size_t>(config.n_audio_layer));
    for (int64_t l = 0; l < config.n_audio_layer; ++l) {
        const std::string p = "model.encoder.layers." + std::to_string(l);
        modules::WhisperEncoderLayerWeights lw;
        lw.attention_norm = binding::norm_from_source(store, source, p + ".self_attn_layer_norm", config.n_audio_state);
        lw.attention.query = linear_weights(store, source, p + ".self_attn.q_proj", config.n_audio_state, config.n_audio_state, true);
        lw.attention.key = linear_weights(store, source, p + ".self_attn.k_proj", config.n_audio_state, config.n_audio_state, false);
        lw.attention.value = linear_weights(store, source, p + ".self_attn.v_proj", config.n_audio_state, config.n_audio_state, true);
        lw.attention.out = linear_weights(store, source, p + ".self_attn.out_proj", config.n_audio_state, config.n_audio_state, true);
        lw.mlp_norm = binding::norm_from_source(store, source, p + ".final_layer_norm", config.n_audio_state);
        lw.mlp.fc1_weight = store.load_tensor(source, p + ".fc1.weight", st, {config.n_audio_state * 4, config.n_audio_state});
        lw.mlp.fc1_bias = store.load_f32_tensor(source, p + ".fc1.bias", {config.n_audio_state * 4});
        lw.mlp.fc2_weight = store.load_tensor(source, p + ".fc2.weight", st, {config.n_audio_state, config.n_audio_state * 4});
        lw.mlp.fc2_bias = store.load_f32_tensor(source, p + ".fc2.bias", {config.n_audio_state});
        weights.layers.push_back(std::move(lw));
    }
    weights.final_norm = binding::norm_from_source(store, source, "model.encoder.layer_norm", config.n_audio_state);
}

// ---------------------------------------------------------------- decoder 权重
// 类型对齐 Canary 的 TransformerDecoderBlockWeights（norm1/self_attention/norm2/cross_attention/norm3/feed_forward），
// 供 TransformerDecoderBlockModule::build_cached_tail 直接使用。

struct DecoderWeights {
    TensorValue embed_tokens;     // [vocab, D] tied lm_head
    TensorValue embed_positions;  // [max_target, D]
    modules::NormWeights final_norm;
    std::vector<modules::TransformerDecoderBlockWeights> layers;
};

// cross packed K/V 权重（qkv_weight/qkv_bias）的存储类型映射：
// 跟随跨注意力 KV 缓存精度（cross_kv_precision_），使权重与激活（kv_memory_）同型，
// 从而通过 ggml-cuda 的 MUL_MAT 类型检查（ggml-cuda.cu:5530：激活 F16 须配权重 F16）。
// 默认 F32 行为不变；F16/BF16 档下 cross packed 权重同步转半精度（对齐 faster-whisper 的 compute_type 一体化）。
assets::TensorStorageType cross_kv_storage_for(WhisperKVStoragePrecision prec) {
    switch (prec) {
        case WhisperKVStoragePrecision::F16:
            return assets::TensorStorageType::F16;
        case WhisperKVStoragePrecision::BF16:
            return assets::TensorStorageType::BF16;
        case WhisperKVStoragePrecision::F32:
        default:
            return assets::TensorStorageType::F32;
    }
}

modules::TransformerDecoderBlockWeights load_block(
    core::BackendWeightStore & store, const assets::TensorSource & source, int64_t layer, int64_t D,
    assets::TensorStorageType cross_kv_storage = assets::TensorStorageType::F32) {
    const std::string p = "model.decoder.layers." + std::to_string(layer) + ".";
    // 单张量权重保留 GGUF 原始存储类型（Native => Q8_0/F16 直接进图由 ggml 反量化），
    // 与 encoder（load_encoder_weights 亦用 Native）一致，避免强制 F32 吃掉量化红利。
    // qkv/cross-qkv 是 CPU 重组打包（make_from_f32）：self-attn qkv 乘的是 decoder 内部 F32 激活，
    //  保持 F32；cross packed K/V 随 cross_kv_storage 转半精度（与 kv_memory_ 同型，见 cross_kv_storage_for）。
    const auto st = assets::TensorStorageType::Native;
    modules::TransformerDecoderBlockWeights w;
    w.norm1 = binding::norm_from_source(store, source, p + "self_attn_layer_norm", D);

    const auto sq = source.require_f32(p + "self_attn.q_proj.weight", {D, D});
    const auto sk = source.require_f32(p + "self_attn.k_proj.weight", {D, D});
    const auto sv = source.require_f32(p + "self_attn.v_proj.weight", {D, D});
    const auto bq = source.require_f32(p + "self_attn.q_proj.bias", {D});
    const auto bk = source.has_tensor(p + "self_attn.k_proj.bias")
                        ? source.require_f32(p + "self_attn.k_proj.bias", {D})
                        : zeros(static_cast<size_t>(D));
    const auto bv = source.require_f32(p + "self_attn.v_proj.bias", {D});
    w.self_attention.qkv_weight = store.make_from_f32(
        TensorShape::from_dims({3 * D, D}), assets::TensorStorageType::F32, pack_qkv(sq, sk, sv));
    w.self_attention.qkv_bias = store.make_from_f32(
        TensorShape::from_dims({3 * D}), assets::TensorStorageType::F32, pack_qkv(bq, bk, bv));
    w.self_attention.out_weight = store.load_tensor(source, p + "self_attn.out_proj.weight",
                                                    st, {D, D});
    w.self_attention.out_bias = store.load_f32_tensor(source, p + "self_attn.out_proj.bias", {D});

    w.norm2 = binding::norm_from_source(store, source, p + "encoder_attn_layer_norm", D);

    const auto cq = source.require_f32(p + "encoder_attn.q_proj.weight", {D, D});
    const auto cbq = source.require_f32(p + "encoder_attn.q_proj.bias", {D});
    const auto ck = source.require_f32(p + "encoder_attn.k_proj.weight", {D, D});
    const auto cv = source.require_f32(p + "encoder_attn.v_proj.weight", {D, D});
    const auto cbk = source.has_tensor(p + "encoder_attn.k_proj.bias")
                         ? source.require_f32(p + "encoder_attn.k_proj.bias", {D})
                         : zeros(static_cast<size_t>(D));
    const auto cbv = source.require_f32(p + "encoder_attn.v_proj.bias", {D});
    w.cross_attention.q_weight = store.make_from_f32(
        TensorShape::from_dims({D, D}), assets::TensorStorageType::F32, cq);
    w.cross_attention.q_bias = store.make_from_f32(
        TensorShape::from_dims({D}), assets::TensorStorageType::F32, cbq);
    // cross packed K/V 权重跟随 cross_kv_storage（默认 F32；F16/BF16 档转半精度与 kv_memory_ 同型）。
    // q_weight/out_weight 乘的是 decoder 内部 F32 激活，保持 F32 不损失精度。
    w.cross_attention.qkv_weight = store.make_from_f32(
        TensorShape::from_dims({2 * D, D}), cross_kv_storage, pack_kv(ck, cv));
    w.cross_attention.qkv_bias = store.make_from_f32(
        TensorShape::from_dims({2 * D}), cross_kv_storage, pack_kv(cbk, cbv));
    w.cross_attention.out_weight = store.load_tensor(source, p + "encoder_attn.out_proj.weight",
                                                     st, {D, D});
    w.cross_attention.out_bias = store.load_f32_tensor(source, p + "encoder_attn.out_proj.bias", {D});

    w.norm3 = binding::norm_from_source(store, source, p + "final_layer_norm", D);

    w.feed_forward.fc1_weight = store.load_tensor(source, p + "fc1.weight", st, {4 * D, D});
    w.feed_forward.fc1_bias = store.load_f32_tensor(source, p + "fc1.bias", {4 * D});
    w.feed_forward.fc2_weight = store.load_tensor(source, p + "fc2.weight", st, {D, 4 * D});
    w.feed_forward.fc2_bias = store.load_f32_tensor(source, p + "fc2.bias", {D});
    return w;
}

DecoderWeights load_decoder_weights(core::BackendWeightStore & store, const assets::TensorSource & source,
                                    const WhisperConfig & cfg,
                                    assets::TensorStorageType cross_kv_storage = assets::TensorStorageType::F32) {
    const int64_t D = cfg.d_model;
    DecoderWeights w;
    // 方案B：嵌入表改 Native 量化（对齐 gemma/qwen）。embed_tokens 同时兼作 tied lm_head，
    // 由 decoder 图内 EmbeddingModule(ggml_get_rows) 查行 + LinearModule(ggml_mul_mat) 输出，
    // 二者均支持 Native 量化权重（同 whisper.cpp 的 tied token_embedding 用法）。
    // 取消 CPU 侧 read_tensor_f32 全表回读（dec_input_ F32 逐行构造）→ 改图内查表，回收约 200MB 常驻并吃到 mmvq。
    w.embed_tokens = store.load_tensor(source, "model.decoder.embed_tokens.weight",
                                       assets::TensorStorageType::Native, {cfg.vocab_size, D});
    w.embed_positions = store.load_tensor(source, "model.decoder.embed_positions.weight",
                                          assets::TensorStorageType::Native, {cfg.max_target_positions, D});
    w.final_norm = binding::norm_from_source(store, source, "model.decoder.layer_norm", D);
    w.layers.reserve(static_cast<size_t>(cfg.decoder_layers));
    for (int64_t i = 0; i < cfg.decoder_layers; ++i) {
        w.layers.push_back(load_block(store, source, i, D, cross_kv_storage));
    }
    return w;
}

// 对 logits 做 log-softmax，取 top-k 的 (token_id, log_prob)。用于 beam search 扩展打分。
// 返回降序（log_prob 从高到低）。k 不超过 vocab 数。
std::vector<std::pair<int32_t, double>> top_k_log_softmax(
    const std::vector<float> & logits,
    int64_t k) {
    if (logits.empty()) {
        return {};
    }
    const double max_ = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    std::vector<double> probs(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(static_cast<double>(logits[i]) - max_);
        sum += probs[i];
    }
    const double log_sum = std::log(sum > 0.0 ? sum : 1.0);
    std::vector<std::pair<int32_t, double>> scored;
    scored.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        scored.emplace_back(static_cast<int32_t>(i),
                            std::log(probs[i]) - log_sum);
    }
    const size_t keep = static_cast<size_t>(k) <= scored.size()
                            ? static_cast<size_t>(k)
                            : scored.size();
    std::partial_sort(
        scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(keep), scored.end(),
        [](const auto & a, const auto & b) { return a.second > b.second; });
    scored.resize(keep);
    return scored;
}

}  // namespace

// ---------------------------------------------------------------- Impl

class WhisperRuntime::Impl {
public:
    Impl(std::shared_ptr<const WhisperWeights> weights, core::ExecutionContext & execution,
         const WhisperRuntimeOptions & options)
        : weights_(std::move(weights)), execution_(&execution),
          cross_kv_precision_(options.cross_kv_precision),
          weight_context_bytes_(options.weight_context_bytes),
          graph_context_bytes_(options.graph_context_bytes) {
        if (weights_ == nullptr || weights_->source == nullptr) {
            throw std::runtime_error("Whisper runtime requires weights");
        }
        const auto & cfg = weights_->config;
        if (cfg.max_target_positions <= 0) {
            throw std::runtime_error("Whisper config missing max_target_positions");
        }
        layout_ = whisper_special_layout(cfg.vocab_size);

        const auto & source = *weights_->source;
        encoder_cfg_.n_mels = cfg.num_mel_bins;
        encoder_cfg_.n_audio_ctx = cfg.max_source_positions;
        encoder_cfg_.n_audio_state = cfg.d_model;
        encoder_cfg_.n_audio_head = cfg.encoder_attention_heads;
        encoder_cfg_.n_audio_layer = cfg.encoder_layers;

        // 两个独立 weight store：encoder 与 decoder 各自的图。
        encoder_store_ = std::make_shared<core::BackendWeightStore>(
            execution.backend(), execution.backend_type(), "whisper.encoder.weights", weight_context_bytes_);
        decoder_store_ = std::make_shared<core::BackendWeightStore>(
            execution.backend(), execution.backend_type(), "whisper.decoder.weights", weight_context_bytes_);

        load_encoder_weights(encoder_embedding_, encoder_cfg_, *encoder_store_, source);
        encoder_store_->upload();

        decoder_weights_ = load_decoder_weights(*decoder_store_, source, cfg,
                                                cross_kv_storage_for(cross_kv_precision_));
        decoder_store_->upload();

        // 方案B：不再做 CPU 侧全表回读（删除 emb_tokens_/emb_positions_ 的 read_tensor_f32）。
        // decoder 输入行改为图内 EmbeddingModule(Native) + ggml_add 查表构造。

        build_encoder_graph();
        build_kv_graphs();

        // 所有权重已通过 encoder_store_/decoder_store_->upload() 上传到 backend(GPU)，
        // 这里释放 weights_->source 持有的 mmap 视图（6.44GB safetensors）。
        // Windows 下 discard_range 是空操作，唯一能真正 Unmap 的是 release_storage()，
        // 与 f5_tts / audio8_asr 等其他模型对齐，消除 CPU 侧权重双份常驻。
        weights_->source->release_storage();
    }

    WhisperTranscriptionResult transcribe(const runtime::AudioBuffer & audio, const WhisperDecodeOptions & opts) {
        const auto & cfg = weights_->config;

        // 1-4) 前端 + encoder + cross-KV 预算 => memory（供 decoder cross-attention）。
        auto memory = encode_to_memory(audio);

        // 5) KV-cache 流式贪心解码
        const int64_t limit = opts.max_len > 0 ? opts.max_len : 64;
        std::vector<int32_t> prompt = build_prompt(opts.language, opts.translate, opts.timestamps);
        // R2 边界 conditioning：当 timestamps 开启且调用方给了片段在全局时间轴的起始偏移时，
        // 在 prompt（<|sot|> lang <|task|> <|timestamp_begin|>）之后追加对应的起始时间戳 token
        // （<|X.XX|>，0.02s/帧），让模型知道"我从第 X 秒中途开始"，抑制跳转片段开头的
        // 边界 artifact（重复上一句尾巴 / 漏半句）。须排在 prompt_len 之前，KV 容量校验才能反映真实长度。
        if (opts.timestamps && opts.clip_offset_seconds > 0.0) {
            const int32_t frame = static_cast<int32_t>(
                std::llround(opts.clip_offset_seconds / 0.02));
            prompt.push_back(layout_.timestamp_begin + frame);
        }
        const int64_t prompt_len = static_cast<int64_t>(prompt.size());
        if (prompt_len - 1 + limit > kv_cache_steps_) {
            throw std::runtime_error("Whisper prompt+generation exceeds KV cache capacity");
        }
        reset_kv_decoding();
        std::vector<int32_t> tokens;
        if (opts.beam_size >= 2) {
            tokens = transcribe_beam_kv(prompt, limit, opts.beam_size, opts.repetition_penalty);
        } else {
            tokens = transcribe_greedy_kv(prompt, limit, opts.repetition_penalty);
        }

        WhisperTranscriptionResult result;
        result.language_token = (prompt.size() >= 2) ? prompt[1] : 0;
        result.token_ids = tokens;
        result.last_logits = last_logits_;
        // 文本只保留非特殊 token（tokenizer::decode 已自动剔除 special/timestamp）。
        result.text = weights_->tokenizer->decode(tokens);
        if (opts.timestamps) {
            result.segments = build_segments(tokens);
        }
        return result;
    }

    // 自动语言检测（官方 detect_language）：仅用 [sot] 单步前向，取语言 token 区间 softmax 的 argmax。
    // 返回 ISO-639 短码（en/zh/ja/...）。
    std::string detect_language(const runtime::AudioBuffer & audio) {
        auto memory = encode_to_memory(audio);
        reset_kv_decoding();
        // 只跑一步：输入 [sot]，得到第一步 logits（存 last_logits_）。
        transcribe_greedy_kv({WhisperTokenIds::sot}, 1);
        // 语言 token 区间：50259(en)..50266(ja)，从第一步 logits 的 softmax 取 argmax。
        static const std::pair<const char *, int32_t> kLangs[] = {
            {"en", 50259}, {"zh", 50260}, {"de", 50261}, {"es", 50262},
            {"ru", 50263}, {"ko", 50264}, {"fr", 50265}, {"ja", 50266},
        };
        const auto & logits = last_logits_;
        if (logits.empty()) {
            throw std::runtime_error("Whisper detect_language: no logits produced");
        }
        int best_idx = 0;
        double best = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < 8; ++i) {
            const int32_t id = kLangs[i].second;
            if (id < 0 || static_cast<size_t>(id) >= logits.size()) {
                continue;
            }
            if (static_cast<double>(logits[static_cast<size_t>(id)]) > best) {
                best = static_cast<double>(logits[static_cast<size_t>(id)]);
                best_idx = static_cast<int>(i);
            }
        }
        return kLangs[static_cast<size_t>(best_idx)].first;
    }

    const WhisperConfig & config() const noexcept { return weights_->config; }

private:
    // 前端（重采样→log-mel）→ encoder 前向 → cross-KV 预算，返回 decoder 的 memory。
    // 供 transcribe 与 detect_language 复用的公共前半段。
    std::vector<float> encode_to_memory(const runtime::AudioBuffer & audio) {
        const auto & cfg = weights_->config;
        const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
            audio.samples, audio.sample_rate, audio.channels, static_cast<int>(kSampleRate));
        const auto log_mel = compute_log_mel(mono, cfg.num_mel_bins);
        const size_t expected_mel = static_cast<size_t>(cfg.num_mel_bins * encoder_cfg_.n_audio_ctx * 2);
        if (log_mel.size() != expected_mel) {
            throw std::runtime_error("Whisper log-mel frame count mismatch: got " +
                                     std::to_string(log_mel.size()) + " expected " + std::to_string(expected_mel));
        }
        core::write_tensor_f32(encoder_input_, log_mel);
        if (core::compute_graph(*execution_, encoder_graph_, encoder_plan_, "WhisperEncoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Whisper encoder compute failed");
        }
        auto memory = core::read_tensor_f32(encoder_output_.tensor);
        const int64_t mem_frames = encoder_cfg_.n_audio_ctx;
        if (static_cast<int64_t>(memory.size()) != mem_frames * cfg.d_model) {
            throw std::runtime_error("Whisper encoder output size mismatch");
        }
        core::write_tensor_float(kv_memory_, memory);
        std::vector<int32_t> mm(static_cast<size_t>(mem_frames), 1);
        core::write_tensor_i32(kv_memory_mask_, mm);
        if (core::compute_graph(*execution_, cross_kv_graph_, cross_kv_plan_, "WhisperCrossKV") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Whisper cross-KV compute failed");
        }
        ggml_backend_synchronize(execution_->backend());
        return memory;
    }
    std::vector<int32_t> build_prompt(const std::string & language, bool translate, bool timestamps) const {
        // 语言 token id 由 openai-whisper 官方 tokenizer 实证（multilingual.tiktoken 追加特殊 token，
        // 自 50259 起按 LANGUAGES 顺序）：en=50259 zh=50260 de=50261 es=50262 ru=50263 ko=50264
        // fr=50265 ja=50266。auto/空/未知语言默认 en（与官方 get_tokenizer 的 language or "en" 一致）。
        int32_t lang_token = 50259;
        std::string lang = language;
        std::transform(lang.begin(), lang.end(), lang.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lang == "zh" || lang == "cmn" || lang == "yue") {
            lang_token = 50260;
        } else if (lang == "de" || lang == "ger" || lang == "deu") {
            lang_token = 50261;
        } else if (lang == "es" || lang == "spa") {
            lang_token = 50262;
        } else if (lang == "ru" || lang == "rus") {
            lang_token = 50263;
        } else if (lang == "ko" || lang == "kor") {
            lang_token = 50264;
        } else if (lang == "fr" || lang == "fra" || lang == "fre") {
            lang_token = 50265;
        } else if (lang == "ja" || lang == "jpn") {
            lang_token = 50266;
        }
        // translate：<|translate|> 替代 <|transcribe|>，输出翻译语言。ID 依语言数量动态（layout_）。
        // timestamps：第 4 个 prompt token 用 <|timestamp_begin|>（代替 <|notimestamps|>），
        //   引导模型从 <|0.00|> 起始并输出时间戳 token（官方 whisper 行为）。
        const int32_t task_token = translate ? layout_.translate : layout_.transcribe;
        const int32_t ending_token = timestamps ? layout_.timestamp_begin : layout_.notimestamps;
        std::vector<int32_t> prompt = {
            WhisperTokenIds::sot,
            lang_token,
            task_token,
            ending_token,
        };
        return prompt;
    }

    void build_encoder_graph() {
        ggml_init_params params{graph_context_bytes_, nullptr, true};
        encoder_ctx_.reset(ggml_init(params));
        if (!encoder_ctx_) {
            throw std::runtime_error("Whisper encoder context allocation failed");
        }
        core::ModuleBuildContext mctx{encoder_ctx_.get(), "whisper.encoder", execution_->backend_type()};
        encoder_input_ = core::make_tensor(
            mctx, GGML_TYPE_F32,
            TensorShape::from_dims({1, encoder_cfg_.n_mels, encoder_cfg_.n_audio_ctx * 2}));
        ggml_set_input(encoder_input_.tensor);
        encoder_output_ = modules::WhisperEmbeddingModule(encoder_cfg_).build(mctx, encoder_input_, encoder_embedding_);
        ggml_set_output(encoder_output_.tensor);
        encoder_graph_ = ggml_new_graph_custom(encoder_ctx_.get(), 65536, false);
        ggml_build_forward_expand(encoder_graph_, encoder_output_.tensor);
        allocate_graph(encoder_ctx_, encoder_graph_, encoder_gallocr_, encoder_plan_, "WhisperEncoder");
    }

    // 正统 Canary 式 KV-cache 解码结构（与 probe_whisper_decoder_cached 完全一致）：
    //   - state ctx 建 memory/x_input/slot/causal_mask/memory_mask + 每层 self K/V cache + cross K/V buffer；
    //   - cross-kv 图把 memory 投影成逐层 cross K/V；
    //   - decoder 图：单 token 输入 x_input -> Nx build_cached_tail -> final LN -> tied lm_head。
    // 一次构图，逐 token compute；cache + cursor 管理跨步 KV。规避全序列每步 O(T^2) 重算。
    void build_kv_graphs() {
        const auto & cfg = weights_->config;
        const int64_t D = cfg.d_model;
        const int64_t heads = cfg.decoder_attention_heads;
        const int64_t head_dim = D / heads;
        const int64_t mem_frames = encoder_cfg_.n_audio_ctx;
        kv_cache_steps_ = cfg.max_target_positions;

        kv_state_ctx_.reset(ggml_init({1024ull * 1024ull, nullptr, true}));
        cross_kv_ctx_.reset(ggml_init({graph_context_bytes_, nullptr, true}));
        dec_ctx_.reset(ggml_init({graph_context_bytes_, nullptr, true}));
        if (!kv_state_ctx_ || !cross_kv_ctx_ || !dec_ctx_) {
            throw std::runtime_error("Whisper KV graph context allocation failed");
        }

        core::ModuleBuildContext sctx{};
        sctx.ggml = kv_state_ctx_.get();
        sctx.backend_type = execution_->backend_type();
        // 整条 cross-attention 链（encoder memory + 投影出的 cross K/V）的常驻精度：
        // 由构造期选项决定（默认 F32，行为不变）。仅 F32/F16 是跨后端通用；
        // BF16 视后端支持（CUDA/Blackwell 支持才用，否则回退 F16 保正确）。
        ggml_type cross_type = GGML_TYPE_F32;
        switch (cross_kv_precision_) {
            case WhisperKVStoragePrecision::F16:
                cross_type = GGML_TYPE_F16;
                break;
            case WhisperKVStoragePrecision::BF16:
                cross_type = GGML_TYPE_BF16;
                break;
            case WhisperKVStoragePrecision::F32:
            default:
                cross_type = GGML_TYPE_F32;
                break;
        }
        // I2：encoder memory 随 cross 链半精度化（它是 cross K/V 投影的上游输入）。
        kv_memory_ = core::make_tensor(sctx, cross_type, TensorShape::from_dims({1, mem_frames, D}));
        // 方案B：decoder 输入由"CPU 填 F32 行"改为"图内查表"。输入改为两个 I32 索引：
        //   dec_token_in_ —— 当前 token id；dec_pos_in_ —— 当前绝对位置。
        // 图内 EmbeddingModule(Native) 按索引 get_rows 得到 token 嵌入与位置嵌入，再 ggml_add 成 [1,1,D]。
        dec_token_in_ = core::make_tensor(sctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        dec_pos_in_ = core::make_tensor(sctx, GGML_TYPE_I32, TensorShape::from_dims({1, 1}));
        dec_slot_ = core::make_tensor(sctx, GGML_TYPE_I32, TensorShape::from_dims({1}));
        dec_causal_mask_ = core::make_tensor(sctx, GGML_TYPE_F16, TensorShape::from_dims({1, kv_cache_steps_}));
        kv_memory_mask_ = core::make_tensor(sctx, GGML_TYPE_I32, TensorShape::from_dims({1, mem_frames}));
        for (size_t layer = 0; layer < decoder_weights_.layers.size(); ++layer) {
            dec_keys_.push_back(core::make_tensor(sctx, GGML_TYPE_F16,
                TensorShape::from_dims({1, kv_cache_steps_, heads, head_dim})));
            dec_values_.push_back(core::make_tensor(sctx, GGML_TYPE_F16,
                TensorShape::from_dims({1, kv_cache_steps_, heads, head_dim})));
            dec_cross_.push_back({
                core::make_tensor(sctx, cross_type,
                    TensorShape::from_dims({1, heads, mem_frames, head_dim})),
                core::make_tensor(sctx, cross_type,
                    TensorShape::from_dims({1, heads, mem_frames, head_dim}))});
        }
        kv_state_buffer_.reset(ggml_backend_alloc_ctx_tensors(kv_state_ctx_.get(), execution_->backend()));
        if (!kv_state_buffer_) {
            throw std::runtime_error("Whisper KV state buffer allocation failed");
        }
        kv_cache_ = runtime::TransformerKVCache(
            kv_cache_steps_, heads * head_dim, dec_keys_, dec_values_, {true, false});

        build_cross_kv_graph(mem_frames);
        build_decoder_kv_graph();
    }

    void build_cross_kv_graph(int64_t mem_frames) {
        core::ModuleBuildContext ctx{};
        ctx.ggml = cross_kv_ctx_.get();
        ctx.backend_type = execution_->backend_type();
        ctx.module_instance_name = "whisper.cross_kv";
        ggml_set_input(kv_memory_.tensor);
        cross_kv_graph_ = ggml_new_graph_custom(cross_kv_ctx_.get(), 65536, false);
        modules::AttentionConfig cross_cfg{weights_->config.d_model, weights_->config.decoder_attention_heads, true};
        cross_cfg.use_packed_kv = true;
        for (size_t i = 0; i < decoder_weights_.layers.size(); ++i) {
            const auto kv = modules::CrossAttentionModule(cross_cfg).build_key_value(
                ctx, kv_memory_, decoder_weights_.layers[i].cross_attention);
            ggml_build_forward_expand(cross_kv_graph_, ggml_cpy(ctx.ggml, kv.key.tensor, dec_cross_[i].key.tensor));
            ggml_build_forward_expand(cross_kv_graph_, ggml_cpy(ctx.ggml, kv.value.tensor, dec_cross_[i].value.tensor));
        }
        allocate_graph(cross_kv_ctx_, cross_kv_graph_, cross_kv_gallocr_, cross_kv_plan_, "WhisperCrossKV");
    }

    void build_decoder_kv_graph() {
        const auto & cfg = weights_->config;
        core::ModuleBuildContext ctx{};
        ctx.ggml = dec_ctx_.get();
        ctx.backend_type = execution_->backend_type();
        ctx.module_instance_name = "whisper.decoder";
        for (auto input : {dec_token_in_, dec_pos_in_, dec_slot_, dec_causal_mask_, kv_memory_mask_}) {
            ggml_set_input(input.tensor);
        }
        dec_graph_ = ggml_new_graph_custom(dec_ctx_.get(), 131072, false);
        // 方案B：输入行改为图内查表。按 dec_token_in_/dec_pos_in_ 两个 I32 索引，
        // 分别 get_rows(Native 量化 embed_tokens/embed_positions) 得到 token/位置嵌入，再相加成 [1,1,D]。
        auto tok_emb = modules::EmbeddingModule({cfg.vocab_size, cfg.d_model})
                           .build(ctx, dec_token_in_, decoder_weights_.embed_tokens);
        auto pos_emb = modules::EmbeddingModule({cfg.max_target_positions, cfg.d_model})
                           .build(ctx, dec_pos_in_, decoder_weights_.embed_positions);
        auto x = core::wrap_tensor(
            ggml_add(ctx.ggml, tok_emb.tensor, pos_emb.tensor),
            TensorShape::from_dims({1, 1, cfg.d_model}),
            GGML_TYPE_F32);
        modules::TransformerDecoderBlockConfig block_cfg{cfg.d_model, cfg.decoder_attention_heads, 4 * cfg.d_model};
        block_cfg.activation = modules::FeedForwardActivation::Gelu;
        block_cfg.use_packed_qkv = true;
        block_cfg.use_packed_kv = true;
        block_cfg.use_flash_cross_attention = false;
        for (size_t i = 0; i < decoder_weights_.layers.size(); ++i) {
            x = modules::TransformerDecoderBlockModule(block_cfg).build_cached_tail(
                ctx, x, decoder_weights_.layers[i],
                dec_keys_[i], dec_values_[i], dec_slot_, dec_causal_mask_, dec_cross_[i], kv_memory_mask_);
        }
        x = modules::LayerNormModule({cfg.d_model}).build(ctx, x, decoder_weights_.final_norm);
        modules::LinearWeights lm;
        lm.weight = decoder_weights_.embed_tokens;
        dec_logits_ = modules::LinearModule({cfg.d_model, cfg.vocab_size, false}).build(ctx, x, lm);
        ggml_set_output(dec_logits_.tensor);
        ggml_build_forward_expand(dec_graph_, dec_logits_.tensor);
        allocate_graph(dec_ctx_, dec_graph_, dec_gallocr_, dec_plan_, "WhisperDecoder");
    }

    void allocate_graph(const std::unique_ptr<ggml_context, GgmlContextDeleter> & /*ctx*/,
                        ggml_cgraph * graph, ggml_gallocr_t & gallocr,
                        core::HostGraphPlan & plan, const char * label) {
        if (execution_->backend_type() == core::BackendType::Cpu) {
            auto opts = engine::runtime::graph_optimization_options_for_backend(
                engine::runtime::GraphOptimizationBackend::Other);
            opts.backend = engine::runtime::GraphOptimizationBackend::Cpu;
            opts.fold_broadcast_repeats = true;
            engine::runtime::optimize_graph(*graph, opts);
        }
        core::validate_backend_graph_supported(execution_->backend(), graph, label);
        gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(execution_->backend()));
        if (!gallocr || !ggml_gallocr_alloc_graph(gallocr, graph)) {
            throw std::runtime_error(std::string(label) + " graph allocation failed");
        }
        core::prepare_host_graph_plan(*execution_, graph, plan);
    }

    void reset_kv_decoding() {
        kv_cache_.clear_on_backend();
        kv_cursor_.reset_to_empty(kv_cache_steps_);
        // 因果掩码：全部 -inf，解码时逐步放行当前 position。
        causal_buf_.assign(static_cast<size_t>(kv_cache_steps_), -std::numeric_limits<float>::infinity());
    }

    // KV-cache 流式贪心：给定 encoder memory（cross K/V 已预算）与 prompt。
    // tokens 数组 = 完整序列（prompt + generated），与绝对 position 精确对应：
    //   - prompt 期（pos < prompt_len）：输入 = tokens[pos] == prompt[pos]，argmax 忽略，不 push；
    //   - 生成期（pos >= prompt_len-1）：输入 = tokens[pos]（已生成的 token），采纳 argmax 并 push，
    //     使 tokens 长度恒 == pos+1，保证 tokens[pos] 无错位。
    std::vector<int32_t> transcribe_greedy_kv(
        const std::vector<int32_t> & prompt, int64_t max_len,
        double repetition_penalty = 1.0) {
        const auto & cfg = weights_->config;
        const int64_t vocab_rows = cfg.vocab_size;
        const int64_t maxp = cfg.max_target_positions;
        const int64_t prompt_len = static_cast<int64_t>(prompt.size());

        std::vector<int32_t> tokens = prompt;   // 完整序列：prompt + generated
        std::vector<float> logits_buf;
        const int64_t total = prompt_len - 1 + max_len;
        const bool apply_penalty = repetition_penalty > 1.0;

        for (int64_t pos = 0; pos < total; ++pos) {
            const auto step = kv_cursor_.next_step();
            const int32_t cache_slot = static_cast<int32_t>(step.cache_slot);

            const int32_t token = tokens[static_cast<size_t>(pos)];
            const int32_t tc = std::max<int32_t>(0, std::min<int32_t>(token, static_cast<int32_t>(vocab_rows) - 1));
            const int32_t pp = std::min<int32_t>(static_cast<int32_t>(pos), static_cast<int32_t>(maxp) - 1);
            // 方案B：只喂 token/position 两个 I32 索引，输入行由 decoder 图内 EmbeddingModule 构造。
            core::write_tensor_i32(dec_token_in_, &tc, 1);
            core::write_tensor_i32(dec_pos_in_, &pp, 1);
            causal_buf_[static_cast<size_t>(pos)] = 0.0F;
            core::write_tensor_i32(dec_slot_, &cache_slot, 1);
            core::write_tensor_f16(dec_causal_mask_, causal_buf_);

            if (core::compute_graph(*execution_, dec_graph_, dec_plan_, "WhisperDecoder") != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("Whisper decoder compute failed");
            }
            ggml_backend_synchronize(execution_->backend());
            kv_cache_.advance_after_direct_append(1);
            kv_cursor_.advance_after_direct_append(1);

            // 生成只从最后一个 prompt 位置（pos == prompt_len-1）之后开始。
            if (pos < prompt_len - 1) {
                continue;   // prompt 期：argmax 忽略，不 push（tokens 已含完整 prompt）。
            }

            core::read_tensor_f32_into(dec_logits_.tensor, logits_buf);
            // P1-2：重复惩罚（OpenAI 语义，>1 抑制重复、保持 logit 符号）。已生成的 token 位于
            // tokens[prompt_len..pos]（含当前输入，它也是上一步已解码的结果）。
            if (apply_penalty) {
                const int64_t start = prompt_len;
                const int64_t end = pos;  // 包含当前已生成输入 token
                for (int64_t gi = start; gi <= end && gi < static_cast<int64_t>(logits_buf.size()); ++gi) {
                    const int32_t t = tokens[static_cast<size_t>(gi)];
                    if (t < 0 || static_cast<int64_t>(t) >= static_cast<int64_t>(logits_buf.size())) {
                        continue;
                    }
                    float & l = logits_buf[static_cast<size_t>(t)];
                    l = l < 0.0F ? l * static_cast<float>(repetition_penalty)
                                 : l / static_cast<float>(repetition_penalty);
                }
            }
            const auto it = std::max_element(logits_buf.begin(), logits_buf.end());
            const int32_t next = static_cast<int32_t>(it - logits_buf.begin());
            if (!std::isfinite(*it)) {
                throw std::runtime_error("Whisper non-finite logits at pos=" + std::to_string(pos));
            }
            last_logits_ = logits_buf;
            tokens.push_back(next);   // push 使 tokens[pos+1] 下一步可用，tokens 长度恒 == pos+1
            if (next == WhisperTokenIds::eot) {
                break;
            }
        }

        std::vector<int32_t> generated(
            tokens.begin() + static_cast<ptrdiff_t>(prompt_len), tokens.end());
        if (!generated.empty() && generated.back() == WhisperTokenIds::eot) {
            generated.pop_back();
        }
        return generated;
    }

    // ---------------------------------------------------------------- beam search
    // 在 KV-cache 流式单 batch 图上实现 beam search，不改框架（不扩展为 batch=B 图）：
    //   - 每个 beam 候选持有自己的独立 KV 状态（TransformerKVState，CPU 快照）；
    //   - 每步对每个未完成候选：import 其 KV -> 单 batch 前向末位 token -> export 出扩展 KV -> top-k 打分；
    //   - 所有候选的扩展并入池，按累计 log-prob 取全局 top beam_size 作为新活跃集。
    // 代价：每步 beam_size 次单 batch 前向 + KV 全量往返（whisper 短序列下可接受）。

    // 单 token 前向到 position 处（position = 该 token 在绝对序列中的下标）。
    // 前向后 kv_cache_ 多 append 一行（覆盖 0..position），读 dec_logits_ 到 out（若非空）。
    void forward_single_position(int32_t token, int64_t position, std::vector<float> * out) {
        const auto & cfg = weights_->config;
        const int64_t vocab_rows = cfg.vocab_size;
        const int64_t maxp = cfg.max_target_positions;
        const int32_t tc = std::max<int32_t>(0, std::min<int32_t>(token, static_cast<int32_t>(vocab_rows) - 1));
        const int32_t pp = static_cast<int32_t>(std::min<int64_t>(position, maxp - 1));
        // 方案B：只喂 token/position 两个 I32 索引，输入行由 decoder 图内 EmbeddingModule 构造。
        core::write_tensor_i32(dec_token_in_, &tc, 1);
        core::write_tensor_i32(dec_pos_in_, &pp, 1);
        // causal：放行 0..position（本次及之前全部 token 可 attend）。
        std::fill(causal_buf_.begin(), causal_buf_.begin() + static_cast<std::ptrdiff_t>(position + 1), 0.0F);
        std::fill(causal_buf_.begin() + static_cast<std::ptrdiff_t>(position + 1), causal_buf_.end(),
                  -std::numeric_limits<float>::infinity());
        const int32_t cache_slot = static_cast<int32_t>(position);  // 非 ring：缓存索引 == position
        core::write_tensor_i32(dec_slot_, &cache_slot, 1);
        core::write_tensor_f16(dec_causal_mask_, causal_buf_);
        if (core::compute_graph(*execution_, dec_graph_, dec_plan_, "WhisperDecoder") != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("Whisper decoder beam compute failed");
        }
        ggml_backend_synchronize(execution_->backend());
        kv_cache_.advance_after_direct_append(1);
        if (out != nullptr) {
            core::read_tensor_f32_into(dec_logits_.tensor, *out);
        }
    }

    struct BeamCandidate {
        std::vector<int32_t> tokens;  // prompt + 已生成（含待前向的末位 back 前的全部）
        double score = 0.0;           // 已生成部分的累计 log-prob
        runtime::TransformerKVState kv;  // 覆盖 tokens[0..len-2]（valid_steps = tokens.size()-1）
        bool done = false;
    };

    std::vector<int32_t> transcribe_beam_kv(
        const std::vector<int32_t> & prompt, int64_t max_len, int64_t beam_size,
        double repetition_penalty = 1.0) {
        const int64_t prompt_len = static_cast<int64_t>(prompt.size());
        const bool apply_penalty = repetition_penalty > 1.0;

        // 建立根候选：把 prompt 前 len-1 个 token 铺进 KV，kv 覆盖 0..len-2。
        reset_kv_decoding();
        for (int64_t pos = 0; pos < prompt_len - 1; ++pos) {
            forward_single_position(prompt[static_cast<size_t>(pos)], pos, nullptr);
        }
        BeamCandidate root;
        root.tokens = prompt;
        root.kv = kv_cache_.export_state();
        std::vector<BeamCandidate> active;
        active.push_back(std::move(root));

        // 完成候选（含 eot）独立收集，避免短完成序列挤占仍在探索的活跃 beam。
        std::vector<BeamCandidate> finished;
        const int64_t vocab_rows = weights_->config.vocab_size;
        std::vector<float> logits;
        logits.reserve(static_cast<size_t>(vocab_rows));

        for (int64_t step = 0; step < max_len && !active.empty(); ++step) {
            std::vector<BeamCandidate> pool;
            for (const auto & cand : active) {
                // import 候选 KV（覆盖 cand.tokens[0..len-2]），下一步 position = len-1。
                kv_cache_.import_state(cand.kv);
                const int64_t next_pos = static_cast<int64_t>(cand.tokens.size()) - 1;
                logits.clear();
                forward_single_position(cand.tokens.back(), next_pos, &logits);
                if (logits.empty()) {
                    throw std::runtime_error("Whisper beam: no logits produced");
                }
                const auto it = std::max_element(logits.begin(), logits.end());
                if (!std::isfinite(*it)) {
                    throw std::runtime_error("Whisper beam: non-finite logits at pos=" +
                                             std::to_string(next_pos));
                }
                last_logits_ = logits;
                // P1-2：重复惩罚（与贪心同语义）。候选已生成 token 位于 cand.tokens[prompt_len..end]。
                if (apply_penalty) {
                    const int64_t start = prompt_len;
                    const int64_t end = static_cast<int64_t>(cand.tokens.size()) - 1;
                    for (int64_t gi = start; gi <= end; ++gi) {
                        const int32_t t = cand.tokens[static_cast<size_t>(gi)];
                        if (t < 0 || static_cast<int64_t>(t) >= static_cast<int64_t>(logits.size())) {
                            continue;
                        }
                        float & l = logits[static_cast<size_t>(t)];
                        l = l < 0.0F ? l * static_cast<float>(repetition_penalty)
                                     : l / static_cast<float>(repetition_penalty);
                    }
                }
                const auto top = top_k_log_softmax(logits, beam_size);
                for (const auto & [tok, lp] : top) {
                    BeamCandidate child;
                    child.tokens = cand.tokens;
                    child.tokens.push_back(tok);
                    child.kv = kv_cache_.export_state();  // 覆盖 cand.tokens 全部（len 行）
                    child.score = cand.score + lp;
                    child.done = (tok == WhisperTokenIds::eot);
                    if (child.done) {
                        finished.push_back(std::move(child));  // 完成序列入池，不再扩展
                    } else {
                        pool.push_back(std::move(child));
                    }
                }
            }

            if (pool.empty()) {
                break;
            }
            std::sort(pool.begin(), pool.end(),
                      [](const BeamCandidate & a, const BeamCandidate & b) { return a.score > b.score; });
            const size_t keep = std::min<size_t>(static_cast<size_t>(beam_size), pool.size());
            active.assign(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(keep));
        }

        // 最终选择：优先已完成（含 eot）候选；否则活跃候选（max_len 截断兜底）。
        // 打分 = 长度归一化(累计 logprob / 生成 token 数) × 去重密度奖励(sqrt(unique_ratio))：
        //   - 归一化防止极短序列(单 token + eot)因累加负项少而天然占优；
        //   - 去重密度奖励抑制复读环(重复短语的平均概率被 sqrt(去重比) 衰减)。
        auto pick_score = [prompt_len](const BeamCandidate & c) -> double {
            const auto gen = static_cast<int64_t>(c.tokens.size()) - prompt_len;
            if (gen <= 0) {
                return -std::numeric_limits<double>::infinity();
            }
            const double base = c.score / static_cast<double>(gen);
            std::vector<int32_t> gen_tokens(
                c.tokens.begin() + static_cast<std::ptrdiff_t>(prompt_len), c.tokens.end());
            std::sort(gen_tokens.begin(), gen_tokens.end());
            const auto unique_end = std::unique(gen_tokens.begin(), gen_tokens.end());
            const double unique_ratio =
                static_cast<double>(static_cast<int64_t>(unique_end - gen_tokens.begin())) / static_cast<double>(gen);
            return base * std::sqrt(std::max(0.0, unique_ratio));
        };
        std::vector<int32_t> final_tokens;
        const std::vector<BeamCandidate> * candidate_pool =
            !finished.empty() ? &finished : &active;
        if (candidate_pool->empty()) {
            return {};
        }
        const BeamCandidate * best_cand = &candidate_pool->front();
        for (const auto & cand : *candidate_pool) {
            if (pick_score(cand) > pick_score(*best_cand)) {
                best_cand = &cand;
            }
        }
        final_tokens = best_cand->tokens;

        std::vector<int32_t> generated(
            final_tokens.begin() + static_cast<ptrdiff_t>(prompt_len), final_tokens.end());
        if (!generated.empty() && generated.back() == WhisperTokenIds::eot) {
            generated.pop_back();
        }
        return generated;
    }

    // 把生成 token 按 whisper 时间戳语义切成字幕段。
    // 时间戳 token：id ∈ [timestamp_begin, timestamp_begin+1500]，帧号 = id - timestamp_begin，
    // 秒 = 帧号 * 0.02（官方 TIME_PRECISION=0.02s）。
    // 官方段结构：<|t_start|> 文本… <|t_end|> <|t_start|> 文本… <|t_end|> …
    //   时间戳出现时：若当前段起点未定、且无累积文本 => 作为段起点；
    //                否则（已有累积文本）=> 作为段终点闭合当前段。
    //   无起始时间戳的文本（音频起点无戳）从 0 秒起成段。末尾未闭合段补一个终点。
    // 把段内文本 token 按 word 切分，并按其 token 数比例把 [t0,t1] 插值分配到每个 word。
    // word 边界：token 原始字节以空格(0x20/tab)开头 → 开新词；否则（英文多 token 单词的
    // 续 token、UTF-8 续字节、CJK 无空格 token）→ 并入前词。CJK 等无空格语言 word 粒度=整段。
    std::vector<WhisperWord> split_segment_words(const std::vector<int32_t> & buf,
                                                 double t0, double t1) const {
        std::vector<WhisperWord> out;
        if (buf.empty() || t1 < t0) {
            return out;
        }
        // 1) 分组成 word（空格开头开新词，否则并入当前词）。
        std::vector<std::vector<int32_t>> word_tokens;
        for (const int32_t id : buf) {
            const std::string b = weights_->tokenizer->token_bytes(id);
            const bool space_lead = !b.empty() &&
                (static_cast<unsigned char>(b[0]) == 0x20 ||
                 static_cast<unsigned char>(b[0]) == 0x09);
            if (word_tokens.empty() || space_lead) {
                word_tokens.emplace_back();
            }
            word_tokens.back().push_back(id);
        }
        // 2) 每 word 权重=token 数，累计比例插值分配 [t0,t1]。
        size_t total = 0;
        for (const auto & wt : word_tokens) {
            total += wt.size();
        }
        double pos = 0.0;
        for (const auto & wt : word_tokens) {
            const double frac_begin = pos / static_cast<double>(total);
            pos += static_cast<double>(wt.size());
            const double frac_end = pos / static_cast<double>(total);
            WhisperWord w;
            w.start_seconds = t0 + (t1 - t0) * frac_begin;
            w.end_seconds = t0 + (t1 - t0) * frac_end;
            w.text = weights_->tokenizer->decode(wt);  // decode 会 trim 该 word 的前导空格
            out.push_back(std::move(w));
        }
        return out;
    }

    std::vector<WhisperSegment> build_segments(const std::vector<int32_t> & tokens) const {
        std::vector<WhisperSegment> out;
        std::vector<int32_t> buf;
        int32_t t_begin = -1;  // 当前段起点的帧号；-1 = 未知
        const int32_t ts_start = layout_.timestamp_begin;
        const int32_t ts_max = ts_start + 1500;

        auto commit = [&](int32_t t_end) {
            if (t_begin < 0) {
                t_begin = 0;   // 无起始时间戳，视作从 0 秒起。
            }
            if (t_end < t_begin) {
                t_end = t_begin;
            }
            WhisperSegment seg;
            seg.start_seconds = static_cast<double>(t_begin) * 0.02;
            seg.end_seconds = static_cast<double>(t_end) * 0.02;
            seg.token_ids = buf;
            seg.text = weights_->tokenizer->decode(buf);
            seg.words = split_segment_words(buf, seg.start_seconds, seg.end_seconds);
            out.push_back(std::move(seg));
            buf.clear();
            t_begin = -1;
        };

        for (const int32_t id : tokens) {
            if (id == WhisperTokenIds::eot) {
                break;
            }
            if (id >= ts_start && id <= ts_max) {
                const int32_t t = id - ts_start;
                if (buf.empty() && t_begin < 0) {
                    t_begin = t;             // 段起点时间戳
                } else {
                    commit(t);               // 段终点时间戳：闭合当前段
                }
            } else if (id < WhisperTokenIds::eot) {
                if (t_begin < 0) {
                    t_begin = 0;             // 文本先于任何起始时间戳，从 0 起段。
                }
                buf.push_back(id);           // 普通文本 token
            }
            // 其它 special（语言/translate/no_speech 等，非时间戳）忽略。
        }
        if (!buf.empty() || t_begin >= 0) {
            commit(t_begin < 0 ? 0 : t_begin);   // 末尾未闭合段补一个终点。
        }
        return out;
    }

    std::shared_ptr<const WhisperWeights> weights_;
    core::ExecutionContext * execution_ = nullptr;
    WhisperKVStoragePrecision cross_kv_precision_ = WhisperKVStoragePrecision::F32;
    // ggml context arena host 提交字节（方案B：由 WhisperRuntimeOptions 注入，默认 128MB/64MB，可配置覆盖）。
    size_t weight_context_bytes_ = kWeightContextBytes;
    size_t graph_context_bytes_ = kGraphContextBytes;

    WhisperSpecialLayout layout_;  // 依 cfg.vocab_size 动态计算的特殊 token 绝对 ID

    modules::WhisperEmbeddingConfig encoder_cfg_;
    modules::WhisperEmbeddingWeights encoder_embedding_;
    std::shared_ptr<core::BackendWeightStore> encoder_store_;
    DecoderWeights decoder_weights_;
    std::shared_ptr<core::BackendWeightStore> decoder_store_;

    // encoder 图
    std::unique_ptr<ggml_context, GgmlContextDeleter> encoder_ctx_;
    ggml_cgraph * encoder_graph_ = nullptr;
    ggml_gallocr_t encoder_gallocr_ = nullptr;
    core::HostGraphPlan encoder_plan_;
    TensorValue encoder_input_;
    TensorValue encoder_output_;

    // KV 状态缓冲区（先构造、后析构，graph 引用其内张量）
    std::unique_ptr<ggml_context, GgmlContextDeleter> kv_state_ctx_;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> kv_state_buffer_{
        nullptr, ggml_backend_buffer_free};
    std::vector<TensorValue> dec_keys_;
    std::vector<TensorValue> dec_values_;
    std::vector<modules::CrossAttentionKeyValue> dec_cross_;
    TensorValue kv_memory_;
    TensorValue kv_memory_mask_;
    TensorValue dec_token_in_;   // 方案B：decoder 输入 token id（I32，图内 EmbeddingModule 查表）
    TensorValue dec_pos_in_;     // 方案B：decoder 输入绝对位置（I32，图内 EmbeddingModule 查位置行）
    TensorValue dec_slot_;
    TensorValue dec_causal_mask_;
    TensorValue dec_logits_;
    int64_t kv_cache_steps_ = 0;
    runtime::TransformerKVCache kv_cache_;
    runtime::BoundedStaticKVDecodeCursor kv_cursor_;
    std::vector<float> causal_buf_;

    // cross-kv 图 & decoder 图（后构造、先析构，确保先于 KV 缓冲区释放）
    std::unique_ptr<ggml_context, GgmlContextDeleter> cross_kv_ctx_;
    ggml_cgraph * cross_kv_graph_ = nullptr;
    ggml_gallocr_t cross_kv_gallocr_ = nullptr;
    core::HostGraphPlan cross_kv_plan_;
    std::unique_ptr<ggml_context, GgmlContextDeleter> dec_ctx_;
    ggml_cgraph * dec_graph_ = nullptr;
    ggml_gallocr_t dec_gallocr_ = nullptr;
    core::HostGraphPlan dec_plan_;

    std::vector<float> last_logits_;
};

// ---------------------------------------------------------------- 公共接口

WhisperRuntime::WhisperRuntime(
    std::shared_ptr<const WhisperWeights> weights,
    core::ExecutionContext & execution_context,
    const WhisperRuntimeOptions & options)
    : impl_(std::make_unique<Impl>(std::move(weights), execution_context, options)) {}

WhisperRuntime::~WhisperRuntime() = default;

const WhisperConfig & WhisperRuntime::config() const noexcept {
    return impl_->config();
}

WhisperTranscriptionResult WhisperRuntime::transcribe(
    const runtime::AudioBuffer & audio,
    const WhisperDecodeOptions & opts) const {
    return impl_->transcribe(audio, opts);
}

std::string WhisperRuntime::detect_language(const runtime::AudioBuffer & audio) const {
    return impl_->detect_language(audio);
}

}  // namespace engine::models::whisper
