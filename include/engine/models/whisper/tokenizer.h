#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::whisper {

// Whisper tokenizer constants (whisper 官方固定值，见详细设计 §5.1)。
// 词表 51865 = GPT-2 字节级 BPE(0..50256) + 追加特殊 token(50257..51864)。
struct WhisperTokenIds {
    static constexpr int32_t eot = 50257;              // <|endoftext|>
    static constexpr int32_t sot = 50258;              // <|startoftranscript|>
    static constexpr int32_t translate = 50358;        // <|translate|>
    static constexpr int32_t transcribe = 50359;       // <|transcribe|>
    static constexpr int32_t nospeech = 50362;         // <|nospeech|>
    static constexpr int32_t notimestamps = 50363;     // <|notimestamps|>
    static constexpr int32_t timestamp_begin = 50364;  // <|0.00|>，之后 1501 个时间戳
    static constexpr int32_t vocab_size = 51865;
};

// Whisper GPT-2 字节级 BPE 解码器。
// 封装 multilingual.tiktoken（<base64(byte-encoded-token)> <rank>）的解析与
// token id -> 可读 UTF-8 文本 的还原；特殊 token 按加入顺序追加在 50257..51864。
class WhisperTokenizer {
public:
    ~WhisperTokenizer();
    WhisperTokenizer(WhisperTokenizer &&) noexcept;
    WhisperTokenizer & operator=(WhisperTokenizer &&) noexcept;
    WhisperTokenizer(const WhisperTokenizer &) = delete;
    WhisperTokenizer & operator=(const WhisperTokenizer &) = delete;

    int32_t base_token_count() const noexcept;   // 50257
    int32_t vocab_size() const noexcept;         // 51865
    std::string decode(const std::vector<int32_t> & ids) const;
    // 返回单个 token id 的原始 UTF-8 字节串（word 切分时判断词首/续字节用）。
    // 特殊/时间戳 token 返回空串。
    std::string token_bytes(int32_t id) const;
    // 是否特殊/时间戳 token（解码时应剔除）
    bool is_special(int32_t id) const noexcept;
    // <|x|> 形式特殊 token 的文本（用于日志/校验），无则返回空
    std::string special_token_text(int32_t id) const;

private:
    struct Impl;
    explicit WhisperTokenizer(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;

    friend std::shared_ptr<WhisperTokenizer> load_whisper_tokenizer(
        const std::filesystem::path & tiktoken_path, int32_t vocab_size);
};

std::shared_ptr<WhisperTokenizer> load_whisper_tokenizer(
    const std::filesystem::path & tiktoken_path, int32_t vocab_size);

}  // namespace engine::models::whisper
