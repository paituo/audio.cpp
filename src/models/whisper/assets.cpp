#include "engine/models/whisper/assets.h"

#include "engine/framework/io/json.h"
#include "engine/framework/model_spec/package.h"

#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace engine::models::whisper {
namespace io = engine::io;
namespace json = engine::io::json;

namespace {

std::filesystem::path spec_path() {
    return engine::model_spec::default_spec_path("whisper");
}

WhisperConfig parse_config(const json::Value & root) {
    WhisperConfig cfg;
    cfg.d_model = json::require_i64(root, "d_model");
    cfg.decoder_attention_heads = json::require_i64(root, "decoder_attention_heads");
    cfg.decoder_layers = json::require_i64(root, "decoder_layers");
    cfg.encoder_attention_heads = json::require_i64(root, "encoder_attention_heads");
    cfg.encoder_layers = json::require_i64(root, "encoder_layers");
    cfg.vocab_size = json::require_i64(root, "vocab_size");
    cfg.max_target_positions = json::require_i64(root, "max_target_positions");
    cfg.num_mel_bins = json::require_i64(root, "num_mel_bins");
    cfg.max_source_positions = json::require_i64(root, "max_source_positions");
    if (cfg.d_model <= 0 || cfg.decoder_attention_heads <= 0 || cfg.decoder_layers <= 0 ||
        cfg.encoder_attention_heads <= 0 || cfg.encoder_layers <= 0 || cfg.vocab_size <= 0 ||
        cfg.max_target_positions <= 0 || cfg.num_mel_bins <= 0 || cfg.max_source_positions <= 0) {
        throw std::runtime_error("invalid Whisper config values");
    }
    if (cfg.d_model % cfg.decoder_attention_heads != 0 || cfg.d_model % cfg.encoder_attention_heads != 0) {
        throw std::runtime_error("Whisper d_model must be divisible by attention head counts");
    }
    // whisper 降级/变体允许更大的词表（如 whisper-ja 的 51866，100 语言——多一个 <|yue|> 独立 token），
    // 但至少要覆盖官方 multilingual 词表（51865），防残缺/错配。
    // ★ 注意：低地址特殊 token 的【绝对 ID 并非固定】——任务/时间戳 token(translate/transcribe/
    //   notimestamps/timestamp_begin) 随语言数量 N=vocab-51766 偏移，runtime 已按 cfg.vocab_size
    //   动态计算（whisper_special_layout），勿再硬编码 99 语言的固定 ID。
    if (cfg.vocab_size < WhisperTokenIds::vocab_size) {
        throw std::runtime_error(
            "Whisper config vocab_size " + std::to_string(cfg.vocab_size) +
            " is smaller than whisper multilingual vocab " +
            std::to_string(WhisperTokenIds::vocab_size));
    }
    return cfg;
}

std::string checkpoint_cache_key(const std::filesystem::path & checkpoint_path) {
    std::error_code ec;
    const auto canonical = std::filesystem::weakly_canonical(checkpoint_path, ec);
    return ec ? checkpoint_path.lexically_normal().string() : canonical.string();
}

std::shared_ptr<const WhisperWeights> load_whisper_weights(
    engine::assets::ResourceBundle resources) {
    auto weights = std::make_shared<WhisperWeights>();
    weights->source = resources.open_tensor_source("weights");
    if (weights->source == nullptr) {
        throw std::runtime_error("whisper weights tensor source could not be opened");
    }
    weights->config = parse_config(resources.parse_json("config"));
    const auto tokenizer_path = resources.require_file("tokenizer");
    weights->tokenizer = load_whisper_tokenizer(
        tokenizer_path, static_cast<int32_t>(weights->config.vocab_size));
    return weights;
}

}  // namespace

std::shared_ptr<const WhisperWeights> load_whisper_weights_cached(
    const std::filesystem::path & model_path) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<const WhisperWeights>> cache;
    auto resources = engine::model_spec::load_resource_bundle(
        model_path,
        engine::model_spec::default_spec_path("whisper"));
    const auto key = checkpoint_cache_key(resources.require_file("weights"));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        if (const auto it = cache.find(key); it != cache.end()) {
            if (auto existing = it->second.lock()) {
                return existing;
            }
        }
    }
    auto loaded = load_whisper_weights(std::move(resources));
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[key] = loaded;
    }
    return loaded;
}

}  // namespace engine::models::whisper
