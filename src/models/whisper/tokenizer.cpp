#include "engine/models/whisper/tokenizer.h"

#include "engine/framework/tokenizers/hf_tokenizer_json.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace engine::models::whisper {
namespace {

// 标准 base64 解码（RFC 4648）。tiktoken 的 token 字段是 <base64(UTF-8 字节编码 unicode)>。
std::string base64_decode(const std::string & in) {
    static const std::string table =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    int val = 0, valb = -8;
    for (const unsigned char c : in) {
        if (c == '=') {
            break;
        }
        const size_t pos = table.find(static_cast<char>(c));
        if (pos == std::string::npos) {
            continue;
        }
        val = (val << 6) + static_cast<int>(pos);
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }
    return out;
}

}  // namespace

struct WhisperTokenizer::Impl {
    // tiktoken 部分 [0..50256]：token = base64 解码后的「字节编码 unicode」串；
    // 追加特殊 token [50257..51864] 占位（字节解码时保持原样）。
    std::vector<std::string> id_to_token;
    // 复用框架 GPT-2 字节级解码器（byte_level + 空 metaspace + trim 前导空格）
    tokenizers::HuggingFaceTokenizerJson decoder;
    int32_t base_count = 0;
    int32_t vocab = 0;

    Impl() : decoder(std::vector<std::string>{}, "", true, true) {}
};

int32_t WhisperTokenizer::base_token_count() const noexcept {
    return impl_ == nullptr ? 0 : impl_->base_count;
}

int32_t WhisperTokenizer::vocab_size() const noexcept {
    return impl_ == nullptr ? 0 : impl_->vocab;
}

WhisperTokenizer::WhisperTokenizer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

WhisperTokenizer::WhisperTokenizer(WhisperTokenizer &&) noexcept = default;
WhisperTokenizer & WhisperTokenizer::operator=(WhisperTokenizer &&) noexcept = default;

WhisperTokenizer::~WhisperTokenizer() = default;

std::string WhisperTokenizer::decode(const std::vector<int32_t> & ids) const {
    if (impl_ == nullptr) {
        return {};
    }
    std::vector<int32_t> text_ids;
    text_ids.reserve(ids.size());
    const int32_t n = static_cast<int32_t>(impl_->id_to_token.size());
    for (const int32_t id : ids) {
        if (id >= 0 && id < n && !is_special(id)) {
            text_ids.push_back(id);  // 只保留真实 BPE 文本 token
        }
    }
    // HuggingFaceTokenizerJson 内部用 id_to_token 表逐 id 拼接 + 字节解码 + 去前导空格
    return impl_->decoder.decode_ids(text_ids);
}

bool WhisperTokenizer::is_special(int32_t id) const noexcept {
    return id >= WhisperTokenIds::eot;  // 50257+ 均为追加特殊/语言/时间戳 token
}

std::string WhisperTokenizer::token_bytes(int32_t id) const {
    if (impl_ == nullptr || id < 0 ||
        static_cast<size_t>(id) >= impl_->id_to_token.size() || is_special(id)) {
        return {};
    }
    // id_to_token 存的是「字节编码 unicode」串：对英文/空格/字节，其字节序列即原始 UTF-8 字节；
    // 对多字节字符，字节编码后的串本身就是合法 UTF-8（业务见 tokenizer.h word 切分说明）。
    return impl_->id_to_token[static_cast<size_t>(id)];
}

std::string WhisperTokenizer::special_token_text(int32_t id) const {
    switch (id) {
        case WhisperTokenIds::eot: return "<|endoftext|>";
        case WhisperTokenIds::sot: return "<|startoftranscript|>";
        case 50358: return "<|translate|>";
        case WhisperTokenIds::transcribe: return "<|transcribe|>";
        case WhisperTokenIds::nospeech: return "<|nospeech|>";
        case WhisperTokenIds::notimestamps: return "<|notimestamps|>";
        default: break;
    }
    return {};
}

std::shared_ptr<WhisperTokenizer> load_whisper_tokenizer(
    const std::filesystem::path & tiktoken_path, int32_t vocab_size) {
    std::ifstream in(tiktoken_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open whisper tiktoken file: " + tiktoken_path.string());
    }
    if (vocab_size < WhisperTokenIds::vocab_size) {
        throw std::runtime_error("whisper tokenizer vocab_size must be at least " +
                                 std::to_string(WhisperTokenIds::vocab_size));
    }
    auto impl = std::make_unique<WhisperTokenizer::Impl>();
    impl->id_to_token.assign(static_cast<size_t>(vocab_size), std::string());

    std::string line;
    int32_t max_id = -1;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        // 行格式：<base64(byte-encoded-token)> <rank>
        const auto sp = line.rfind(' ');
        if (sp == std::string::npos) {
            continue;
        }
        const std::string token_b64 = line.substr(0, sp);
        int32_t rank = 0;
        try {
            rank = std::stoi(line.substr(sp + 1));
        } catch (...) {
            continue;
        }
        if (rank < 0 || rank >= vocab_size) {
            continue;
        }
        impl->id_to_token[static_cast<size_t>(rank)] = base64_decode(token_b64);
        max_id = std::max(max_id, rank);
    }
    if (max_id < 0) {
        throw std::runtime_error("whisper tiktoken file contained no tokens: " + tiktoken_path.string());
    }
    // 把表交给解码器；byte_level=true 会把「字节编码 unicode」还原为真字节
    impl->decoder = tokenizers::HuggingFaceTokenizerJson(impl->id_to_token, "", true, true);
    impl->base_count = max_id + 1;
    impl->vocab = vocab_size;
    // make_shared 需要访问私有构造函数（friend 仅覆盖本函数体内直接调用），
    // 因此用显式 new + shared_ptr 构造。
    return std::shared_ptr<WhisperTokenizer>(new WhisperTokenizer(std::move(impl)));
}

}  // namespace engine::models::whisper
