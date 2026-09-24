#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/models/whisper/tokenizer.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace engine::models::whisper {

// Whisper 架构元数据（从 model 目录的 config.json 解析）。
// 与 openai/whisper HF config 字段对应：d_model、decoder/encoder_attention_heads、
// decoder/encoder_layers、vocab_size、max_target_positions、num_mel_bins、max_source_positions。
struct WhisperConfig {
    int64_t d_model = 0;                 // 共享隐藏维度（whisper-tiny=384）
    int64_t decoder_attention_heads = 0; // whisper-tiny=6
    int64_t decoder_layers = 0;          // whisper-tiny=4
    int64_t encoder_attention_heads = 0; // whisper-tiny=6
    int64_t encoder_layers = 0;          // whisper-tiny=4
    int64_t vocab_size = 0;              // 51865
    int64_t max_target_positions = 0;    // 448（decoder 位置嵌入上限）
    int64_t num_mel_bins = 0;            // 80
    int64_t max_source_positions = 0;    // 1500（encoder 上下文 / 30s）
};

// Whisper 家族权重句柄。只持 tensor 源 + 配置 + 词表；
// 具体 encoder/decoder 的后端权重在 runtime 构图时才从 source 加载
// （encoder 与 decoder 各用独立 BackendWeightStore，符合本家族多图结构）。
struct WhisperWeights {
    std::shared_ptr<const assets::TensorSource> source;
    WhisperConfig config;
    std::shared_ptr<WhisperTokenizer> tokenizer;
};

// 按 model 目录缓存加载 WhisperWeights（mutex + weak_ptr 缓存模式，见 citrinet assets）。
std::shared_ptr<const WhisperWeights> load_whisper_weights_cached(
    const std::filesystem::path & model_path);

}  // namespace engine::models::whisper
