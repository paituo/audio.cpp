#pragma once

#include "engine/framework/core/execution_context.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/whisper/assets.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace engine::models::whisper {

// word 级时间戳（word_timestamps 模式）：段内每个词一条，起止秒由段内 token
// 数比例在 [段start, 段end] 上插值分配（openai-whisper 无注意力 fallback）。
struct WhisperWord {
    double start_seconds = 0.0;
    double end_seconds = 0.0;
    std::string text;
};

// 单个带时间戳的字幕段（timestamps 模式下按 <|t|> token 切分）。
// start/end 为 whisper 时间戳帧（1 帧 = 20ms）换算出的秒。
struct WhisperSegment {
    double start_seconds = 0.0;
    double end_seconds = 0.0;
    std::string text;
    std::vector<int32_t> token_ids;   // 段内文本 token（不含时间戳）
    std::vector<WhisperWord> words;   // 段内 word 级时间戳（word_timestamps）；空 = 未启用
};

// Whisper 转写结果：文本 + 生成 token 序列（不含 prompt）+ 末位 logits（诊断用）+ 时间戳段。
struct WhisperTranscriptionResult {
    std::string text;
    int32_t language_token = 0;
    std::vector<int32_t> token_ids;        // 全生成 token（含时间戳 token）
    std::vector<float> last_logits;
    std::vector<WhisperSegment> segments;  // timestamps 模式按时间戳切分；否则为空
};

// cross-attention 用的 encoder memory K/V（dec_cross_）在 GPU 常驻 buffer 的存储精度。
// 设计对齐 faster-whisper `--compute_type` 的"运行期可选"语义，而非把精度焊死在源码里：
//   F32 —— 默认，与历史行为完全一致（保守保精度，改动零风险）；
//   F16 —— 半精度存储，cross K/V 显存直接减半（large-v3 下 −245.8 MB/份），报损风险低（仅前向 matmul）；
//   BF16 —— 半精度bfloat（如后端 GPU 支持则更贴合 compute_type=auto 的择优结果）。
enum class WhisperKVStoragePrecision {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
};

// whisper 运行时构造选项（构图期生效：buffer 类型在 build_kv_graphs 时确定）。
// 第二个参数带默认值，旧调用方（仅传 weights + execution）无需改动、行为不变。
struct WhisperRuntimeOptions {
    WhisperKVStoragePrecision cross_kv_precision = WhisperKVStoragePrecision::F32;
    // ggml 图/权重 context arena 的 host 提交字节（ggml_init 按 mem_size 全量对齐提交；whisper 实际只放元数据 <10MB）。
    // 默认 128MB/64MB（原 kWeightContextBytes/kGraphContextBytes 常量），可通过 whisper.weight_context_mb /
    // whisper.graph_arena_mb 会话选项覆盖（走框架 parse_size_mb_option，对齐 hviske/fun_asr 等模型的显存可调模式）。
    size_t weight_context_bytes = 128ull * 1024ull * 1024ull;
    size_t graph_context_bytes = 64ull * 1024ull * 1024ull;
};

// whisper 解码选项（task / 时间戳 / 语言 / 长度 / beam）。
struct WhisperDecodeOptions {
    bool translate = false;    // true => prompt 用 <|translate|>，输出目标语言（默认为英语）
    bool timestamps = false;   // true => 不用 <|notimestamps|>，解析时间戳 token 出段
    std::string language = "en";
    int64_t max_len = 0;       // 0 => 参考默认（64）
    int64_t beam_size = 0;     // 0 或 1 => 贪心（KV-cache 流式）；>=2 => beam search（逐候选单 batch 前向 + KV state 往返）
    double repetition_penalty = 1.0;  // >=1.0。1.0 => 不惩罚（默认）；>1.0 => 抑制已生成 token 的重复（对 logits 已生成区间扣减）
    double clip_offset_seconds = 0.0;  // >0 && timestamps：向 prompt 追加起始时间戳 token（边界 conditioning）
};

// whisper 家族运行时：
//   - 前端：WhisperLogMelExtractor 生成 log-mel；
//   - encoder：WhisperEmbeddingModule（HF 权重命名，head 从 config 解析，规避共享 head=12 bug），
//     输出 encoder 隐藏态作为 decoder 的 memory（经 CPU 中转，与 A3 探针全序列路径一致）；
//   - decoder：全序列 build（非 build_cached_tail），逐 token 贪心，SOT+语言+transcribe+notimestamps 起步。
// 首版保证正确性（可复现 [Music]）；KV-cache 流式留待后续独立优化。
class WhisperRuntime {
public:
    WhisperRuntime(
        std::shared_ptr<const WhisperWeights> weights,
        core::ExecutionContext & execution_context,
        const WhisperRuntimeOptions & options = {});
    ~WhisperRuntime();

    // 阻塞转写整段音频（≤30s；更长请在外层分段）。
    // opts.translate：true 时输出为翻译语言（openai-whisper 语义下默认英语）。
    // opts.timestamps：true 时结果携带按时间戳切分出的 segments。
    WhisperTranscriptionResult transcribe(
        const runtime::AudioBuffer & audio,
        const WhisperDecodeOptions & opts) const;

    // 自动语言检测：对音频做单步前向，返回最可能的 ISO-639 短码（en/zh/ja/...）。
    std::string detect_language(const runtime::AudioBuffer & audio) const;

    const WhisperConfig & config() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace engine::models::whisper
