#include "engine/models/whisper/session.h"

#include "engine/framework/audio/chunking.h"
#include "engine/framework/audio/conversion.h"
#include "engine/framework/model_spec/package.h"
#include "engine/framework/debug/profiler.h"
#include "engine/framework/debug/trace.h"
#include "engine/framework/runtime/options.h"
#include "engine/models/silero_vad/session.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace engine::models::whisper {
namespace {

std::filesystem::path spec_path() {
    return engine::model_spec::default_spec_path("whisper");
}

// 内部 VAD（silero_vad）默认权重目录：assets/framework/models/silero_vad。
std::filesystem::path default_vad_model_path() {
    return std::filesystem::path("assets") / "framework" / "models" / "silero_vad";
}

std::string language_from_options(const std::unordered_map<std::string, std::string> & options) {
    const auto it = options.find("language");
    if (it == options.end() || it->second.empty()) {
        return "en";
    }
    return it->second;
}

int64_t max_len_from_options(const std::unordered_map<std::string, std::string> & options) {
    const auto it = options.find("max_tokens");
    if (it == options.end() || it->second.empty()) {
        return 0;
    }
    return std::stoll(it->second);
}

bool translate_from_options(const std::unordered_map<std::string, std::string> & options) {
    const auto it = options.find("task");
    if (it != options.end() && it->second == "translate") {
        return true;
    }
    return false;
}

bool timestamps_from_options(const std::unordered_map<std::string, std::string> & options) {
    const auto it = options.find("timestamps");
    if (it == options.end() || it->second.empty()) {
        return false;
    }
    return it->second == "1" || it->second == "true" || it->second == "yes";
}

// 读取片段在全局时间轴的起始偏移（秒）。兼容 OpenAI clip_timestamps 语义与单 float 偏移：
//   clip_timestamps     = "[start,end]" 或 "[start,null]"（取 start）
//   clip_offset_seconds = "125.4"        （单 float，仅起始偏移）
// 缺省 / 无法解析 => 0.0（完全回退当前行为：不注入起始时间戳 token、时间戳相对片段起点）。
// 注：input options 已在 run()/streaming 入口经 normalize_request_options 归一化（whisper. 前缀已剥去）。
double clip_offset_from_options(const std::unordered_map<std::string, std::string> & options) {
    if (auto it = options.find("clip_timestamps"); it != options.end() && !it->second.empty()) {
        const std::string & s = it->second;
        const size_t c0 = s.find('[');
        const size_t c1 = s.find(',');
        if (c0 != std::string::npos && c1 != std::string::npos) {
            try {
                const double v = std::stod(s.substr(c0 + 1, c1 - c0 - 1));
                return v > 0.0 ? v : 0.0;
            } catch (...) { /* fallthrough */ }
        }
    }
    if (auto it = options.find("clip_offset_seconds"); it != options.end() && !it->second.empty()) {
        try {
            const double v = std::stod(it->second);
            return v > 0.0 ? v : 0.0;
        } catch (...) { /* fallthrough */ }
    }
    return 0.0;
}

int64_t beam_size_from_options(const std::unordered_map<std::string, std::string> & options) {
    const auto it = options.find("beam_size");
    if (it == options.end() || it->second.empty()) {
        return 0;
    }
    const int64_t value = std::stoll(it->second);
    return value >= 2 ? value : 0;
}

// 重复惩罚：裸名 repetition_penalty。缺省/非法 => 1.0（不惩罚，与历史贪心行为一致）。
// 实现侧 >1.0 才启用，<1.0（鼓励重复）不支持，钳到 1.0。
double repetition_penalty_from_options(
    const std::unordered_map<std::string, std::string> & options) {
    const auto it = options.find("repetition_penalty");
    if (it == options.end() || it->second.empty()) {
        return 1.0;
    }
    try {
        const double value = std::stod(it->second);
        return value > 1.0 ? value : 1.0;
    } catch (const std::exception &) {
        return 1.0;
    }
}

// cross-attention 链（kv_memory_ + dec_cross_）的常驻精度。
// 空/未知 => F32（默认保底，与历史行为一致）；f16/bf16 => 半精度省显存。
// 对齐 faster-whisper compute_type 的"运行期可配"语义。C++17 上手写构造（不用指定初始化器）。
// 注意：本函数在构造器内调用，读取的是 SessionOptions（session 作用域，带 whisper. 前缀），
// 与 request 作用域的 *_from_options（裸名）不是同一路。
WhisperRuntimeOptions whisper_runtime_options_from_options(
    const std::unordered_map<std::string, std::string> & options) {
    WhisperRuntimeOptions ropts;
    const auto it = options.find("whisper.cross_kv_precision");
    if (it != options.end() && !it->second.empty()) {
        const std::string & v = it->second;
        if (v == "f16" || v == "fp16" || v == "float16") {
            ropts.cross_kv_precision = WhisperKVStoragePrecision::F16;
        } else if (v == "bf16" || v == "bfloat16") {
            ropts.cross_kv_precision = WhisperKVStoragePrecision::BF16;
        }
        // 其余（含 f32/auto/未知）保持默认 F32。
    }
    // 方案B：ggml context arena 显存可调（对齐 hviske/fun_asr 等模型的 weight_context/graph_arena 模式）。
    // 键名 whisper.weight_context_mb / whisper.graph_arena_mb，缺省回落到 runtime.h 默认（128MB/64MB）。
    ropts.weight_context_bytes = engine::runtime::parse_size_mb_option(
        options, {"whisper.weight_context_mb"}, ropts.weight_context_bytes);
    ropts.graph_context_bytes = engine::runtime::parse_size_mb_option(
        options, {"whisper.graph_arena_mb"}, ropts.graph_context_bytes);
    return ropts;
}

// 统一 request 选项前置归一化：历史调用方可能用 "whisper.<name>" 前缀键，而框架/spec 契约
// 声明的是裸名。这里把已知 whiser 前缀键剥成裸名（向后兼容），使下游 *_from_options 只读裸名即可。
// 附带迁移旧别名（audio_chunk_mode=quiet_energy 等价 auto）。
// P2-3 诊断：对带 "whisper." 前缀、但整名不在已知 whisper 选项白名单内的键输出告警——
// 这是「whisper 专属选项拼错/未声明却被静默忽略」的最直接信号。不带前缀的框架通用键不告警。
std::unordered_map<std::string, std::string> normalize_request_options(
    std::unordered_map<std::string, std::string> options) {
    constexpr std::string_view kPrefix = "whisper.";
    // 已知 whisper 消费的裸名（spec request/session + *_from_options + 构造器读取）。
    // 不带前缀的框架通用键（route/audio_chunk_seconds 等）不属于 whisper 专属，不在此列、也不告警。
    static const std::unordered_set<std::string> kKnownBareNames = {
        "language", "max_tokens", "task", "timestamps",
        "beam_size", "repetition_penalty",
        "clip_offset_seconds", "clip_timestamps",
        "audio_chunk_mode", "audio_chunk_duration_sec",
        "vad_model_path", "cross_kv_precision",
        "weight_context_mb", "graph_arena_mb",
    };
    if (engine::debug::log_enabled()) {
        for (const auto & entry : options) {
            const std::string & raw_key = entry.first;
            if (raw_key.rfind(kPrefix, 0) != 0) {
                continue;  // 非 whisper 前缀：框架通用键，不诊断
            }
            const std::string bare = raw_key.substr(kPrefix.size());
            if (kKnownBareNames.find(bare) == kKnownBareNames.end()) {
                engine::debug::log_message(
                    "[WHISPER_OPTION] unrecognized whisper option '" + raw_key +
                    "' (name '" + bare + "') will be ignored; check spelling or add to model_specs/whisper.json");
            }
        }
    }
    std::unordered_map<std::string, std::string> out;
    out.reserve(options.size());
    for (auto & [key, value] : options) {
        std::string normalized = key;
        if (normalized.rfind(kPrefix, 0) == 0) {
            normalized.erase(0, kPrefix.size());
        }
        // 裸名优先：若裸名与剥前缀后的键都出现，取显式裸名值。
        out[std::move(normalized)] = std::move(value);
        if (normalized == "audio_chunk_mode" && (value == "quiet_energy" || value == "default")) {
            out[normalized] = "auto";
        }
    }
    return out;
}

// 长音频分段模式：委托框架统一契约 engine::audio::parse_audio_chunk_mode，
// 支持 auto/fixed/quiet_energy->auto/vad/none，并解析 audio_chunk_duration_sec 等别名窗长。
// input options 已在入口归一化（裸名）。
engine::audio::AudioChunkMode chunk_mode_from_options(
    const std::unordered_map<std::string, std::string> & options) {
    return engine::audio::parse_audio_chunk_mode(options);
}

// 把 whisper 解码结果映射到框架 TaskResult：text_output + （可选）speech_segments。
// output_language：转写结果的语言（translate 时为目标语言 "en"，否则为源语言）。
// start_sample_offset：本段在全局 16k mono 中的起始样本（长音频分段用），把 whisper 段内相对
// 时间平移成全局时间。
void populate_result(
    runtime::TaskResult & result,
    const engine::models::whisper::WhisperTranscriptionResult & transcription,
    const std::string & output_language,
    int64_t start_sample_offset) {
    if (!transcription.segments.empty()) {
        result.speech_segments.reserve(result.speech_segments.size() + transcription.segments.size());
        for (const auto & seg : transcription.segments) {
            runtime::SpeechSegment out;
            out.span.start_sample = start_sample_offset + static_cast<int64_t>(seg.start_seconds * 16000.0);
            out.span.end_sample = start_sample_offset + static_cast<int64_t>(seg.end_seconds * 16000.0);
            out.confidence = 1.0f;
            out.text = seg.text;
            result.speech_segments.push_back(std::move(out));

            // word 级时间戳：把段内 word（秒）平移成全局样本，累加进 result.word_timestamps。
            for (const auto & w : seg.words) {
                runtime::WordTimestamp wt;
                wt.span.start_sample =
                    start_sample_offset + static_cast<int64_t>(w.start_seconds * 16000.0);
                wt.span.end_sample =
                    start_sample_offset + static_cast<int64_t>(w.end_seconds * 16000.0);
                wt.word = w.text;
                wt.confidence = 1.0f;
                result.word_timestamps.push_back(std::move(wt));
            }
        }
    }
    if (result.text_output.has_value()) {
        // 长音频分段：累积拼接文本。
        if (!transcription.text.empty()) {
            if (!result.text_output->text.empty()) {
                result.text_output->text += " ";
            }
            result.text_output->text += transcription.text;
        }
    } else {
        result.text_output = runtime::Transcript{transcription.text, output_language};
    }
}

class WhisperASRLoader final : public runtime::IVoiceModelLoader {
public:
    std::string family() const override {
        return "whisper";
    }

    runtime::CapabilitySet advertised_capabilities() const override {
        runtime::CapabilitySet out;
        out.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        out.supports_timestamps = true;
        return out;
    }

    bool can_load(const runtime::ModelLoadRequest & request) const override {
        if (request.family_hint.has_value() && *request.family_hint != family()) {
            return false;
        }
        try {
            (void) engine::model_spec::load_resource_bundle(request.model_path, spec_path());
            return true;
        } catch (...) {
            return false;
        }
    }

    runtime::ModelInspection inspect(const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::runtime_error("Whisper does not expose selectable config assets");
        }
        const auto resources = engine::model_spec::load_resource_bundle(
            request.model_path,
            spec_path());
        const auto & weight_path = resources.require_file("weights");
        runtime::ModelInspection inspection;
        inspection.model_root = resources.model_root();
        inspection.metadata.family = family();
        inspection.metadata.variant = weight_path.stem().string();
        inspection.metadata.description = "Whisper multilingual ASR loaded from local tensor assets.";
        inspection.capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        inspection.capabilities.supports_timestamps = true;
        inspection.discovered_configs = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path(),
            engine::model_spec::ResourceKind::Files);
        inspection.discovered_weights = runtime::discover_named_assets_from_package_spec(
            request.model_path,
            spec_path(),
            engine::model_spec::ResourceKind::Tensors);
        return inspection;
    }

    std::unique_ptr<runtime::ILoadedVoiceModel> load(const runtime::ModelLoadRequest & request) const override {
        if (request.config_id.has_value()) {
            throw std::runtime_error("Whisper does not expose selectable config assets");
        }
        const auto resources = engine::model_spec::load_resource_bundle(
            request.model_path,
            spec_path());
        const auto & weight_path = resources.require_file("weights");
        runtime::ModelMetadata metadata;
        metadata.family = family();
        metadata.variant = weight_path.stem().string();
        metadata.description = "Whisper multilingual ASR loaded from local tensor assets.";
        runtime::CapabilitySet capabilities;
        capabilities.supported_tasks = {
            {runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline, runtime::RunMode::Streaming}},
        };
        capabilities.supports_timestamps = true;
        return std::make_unique<WhisperASRLoadedModel>(
            std::move(metadata),
            std::move(capabilities),
            load_whisper_weights_cached(request.model_path));
    }
};

}  // namespace

WhisperASRSession::WhisperASRSession(
    runtime::TaskSpec task,
    runtime::SessionOptions options,
    std::shared_ptr<const WhisperWeights> weights)
    : RuntimeSessionBase(options),
      task_(std::move(task)),
      runtime_(std::move(weights), execution_context(),
               whisper_runtime_options_from_options(options.options)) {
    // 必须在构造函数体内读取 this->options()：base 已构造完成，完整持有 SessionOptions。
    // 注意：参数名 `options` 会遮蔽成员函数 options()，需显式 this-> 解除遮蔽。
    vad_model_path_ = runtime::find_option(this->options().options, {"whisper.vad_model_path"})
                          .value_or(default_vad_model_path().string());
    if (task_.task != runtime::VoiceTaskKind::Asr) {
        throw std::runtime_error("Whisper only supports VoiceTaskKind::Asr");
    }
    if (task_.mode != runtime::RunMode::Offline && task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Whisper supports offline and streaming sessions");
    }
}

WhisperASRSession::~WhisperASRSession() = default;

std::string WhisperASRSession::family() const {
    return "whisper";
}

runtime::VoiceTaskKind WhisperASRSession::task_kind() const {
    return task_.task;
}

runtime::RunMode WhisperASRSession::run_mode() const {
    return task_.mode;
}

void WhisperASRSession::prepare(const runtime::SessionPreparationRequest & request) {
    if (!request.audio.has_value()) {
        throw std::runtime_error("Whisper prepare() requires an audio contract");
    }
    mark_prepared();
}

runtime::TaskResult WhisperASRSession::run(const runtime::TaskRequest & request) {
    require_prepared("Whisper run()");
    if (!request.audio_input.has_value()) {
        throw std::runtime_error("Whisper run() requires audio_input");
    }
    const auto wall_start = std::chrono::steady_clock::now();
    // 统一 request 选项归一化（剥 whisper. 前缀 → 裸名，迁移旧别名），供下游 *_from_options 读取。
    const auto options = normalize_request_options(request.options);
    // 语言：显式给定则用之；"auto" 触发自动检测（对整段做单步前向）。
    std::string language = language_from_options(options);
    if (language == "auto") {
        language = runtime_.detect_language(*request.audio_input);
    }
    engine::models::whisper::WhisperDecodeOptions opts;
    opts.language = language;
    opts.max_len = max_len_from_options(options);
    opts.translate = translate_from_options(options);
    opts.timestamps = timestamps_from_options(options);
    opts.beam_size = beam_size_from_options(options);
    opts.repetition_penalty = repetition_penalty_from_options(options);
    opts.clip_offset_seconds = clip_offset_from_options(options);
    // translate 语义（openai-whisper）：输出固定为英语。
    const std::string output_language = opts.translate ? "en" : language;

    // whisper 前端需 16k mono；先统一重采样一次，超长时按此切段。
    constexpr int64_t kSampleRate = 16000;
    constexpr int64_t kMaxWindowSamples = 480000;  // 30s @16k
    const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
        request.audio_input->samples,
        request.audio_input->sample_rate,
        request.audio_input->channels,
        static_cast<int>(kSampleRate));

    runtime::TaskResult result;
    auto transcribe_window = [&](const std::vector<float> & window, int64_t start_sample) {
        runtime::AudioBuffer buf;
        buf.sample_rate = static_cast<int>(kSampleRate);
        buf.channels = 1;
        buf.samples = window;
        const auto transcription = runtime_.transcribe(buf, opts);
        populate_result(result, transcription, output_language, start_sample);
        engine::debug::trace_log_scalar(
            "whisper.chunk_generated_tokens", static_cast<int64_t>(transcription.token_ids.size()));
    };

    const int64_t total = static_cast<int64_t>(mono.size());
    // R1/R3：片段在全局时间轴的起始偏移（采样数）。单段时作为绝对对齐基准；
    // 长音频分段时累加到每个子窗口（R3），使时间戳仍落在全局绝对坐标上。
    const int64_t clip_offset_samples =
        static_cast<int64_t>(std::llround(opts.clip_offset_seconds * kSampleRate));
    if (total <= kMaxWindowSamples) {
        transcribe_window(mono, clip_offset_samples);
    } else {
        // 长音频：按分段策略切成活动段（每段 ≤30s），逐段转写并平移全局时间戳。
        //   委托框架统一契约 engine::audio，支持 auto(=quiet_energy)/vad/fixed/none。
        std::vector<runtime::TimeSpan> spans;
        const engine::audio::AudioChunkMode chunk_mode = chunk_mode_from_options(options);
        if (chunk_mode == engine::audio::AudioChunkMode::Vad) {
            runtime::AudioBuffer mono_buf;
            mono_buf.sample_rate = static_cast<int>(kSampleRate);
            mono_buf.channels = 1;
            mono_buf.samples = mono;
            const engine::audio::VadAudioChunkOptions vopts{
                kMaxWindowSamples,   // max_chunk_samples：每段 ≤30s
                kSampleRate,         // merge_gap_samples：1s 内合并相邻说话段
                kSampleRate / 2,     // padding_samples：0.5s 语音前后 padding
            };
            spans = engine::audio::plan_vad_audio_chunks(mono_buf, vad_session(), vopts);
        } else if (chunk_mode == engine::audio::AudioChunkMode::Fixed) {
            // 定长窗切分：按 audio_chunk_duration_sec（默认 30s）等分，无智能切分。
            const float fixed_sec = engine::audio::parse_audio_chunk_seconds_override(options)
                                        .value_or(30.0f);
            const int64_t window_samples =
                static_cast<int64_t>(std::max(1.0f, fixed_sec) * static_cast<float>(kSampleRate));
            for (int64_t begin = 0; begin < total; begin += window_samples) {
                const int64_t end = std::min(begin + window_samples, total);
                if (end > begin) {
                    spans.push_back(runtime::TimeSpan{begin, end});
                }
            }
        } else {
            // auto / quiet_energy / none（whisper 单窗口上限 30s，超长音频必须分段）：
            // 统一走纯能量谷切分，零 VAD 依赖，保证覆盖完整音频。
            const engine::audio::QuietEnergyAudioChunkOptions chunk_opts{
                kMaxWindowSamples,      // chunk_samples：每段最大 30s
                kSampleRate * 2,        // boundary_context_samples：2s 内找能量谷
                kSampleRate / 2,        // min_energy_window_samples：0.5s 能量窗口
            };
            spans = engine::audio::plan_quiet_energy_audio_chunks(mono, chunk_opts);
        }
        if (spans.empty()) {
            transcribe_window(mono, clip_offset_samples);  // 兜底：切分失败则整段（可能被截断）
        } else {
            for (const auto & span : spans) {
                const int64_t begin = span.start_sample;
                const int64_t end = std::min(span.end_sample, total);
                if (end <= begin) {
                    continue;
                }
                std::vector<float> window(
                    mono.begin() + static_cast<std::ptrdiff_t>(begin),
                    mono.begin() + static_cast<std::ptrdiff_t>(end));
                // R3：外部偏移累加到子窗口起点，子窗口时间戳落在全局绝对坐标上。
                transcribe_window(window, clip_offset_samples + begin);
            }
        }
    }

    engine::debug::timing_log_scalar("whisper.session.wall_ms", engine::debug::elapsed_ms(wall_start));
    return result;
}

runtime::StreamingPolicy WhisperASRSession::streaming_policy() const {
    runtime::StreamingPolicy policy;
    policy.input = runtime::StreamingInputKind::AudioChunks;
    policy.output = runtime::StreamingOutputKind::FinalResult;
    policy.preferred_audio_chunk_seconds = 1.0;
    return policy;
}

void WhisperASRSession::start_stream(const runtime::TaskRequest & request) {
    require_prepared("Whisper start_stream()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Whisper start_stream() requires a streaming session");
    }
    reset();
    streaming_request_ = request;
    if (streaming_request_.audio_input.has_value()) {
        streaming_request_.audio_input->samples.clear();
    }
    stream_started_ = true;
}

void WhisperASRSession::set_stream_event_sink(runtime::StreamEventCallback sink) {
    stream_event_sink_ = std::move(sink);
}

void WhisperASRSession::reset() {
    require_prepared("Whisper reset()");
    streaming_request_ = runtime::TaskRequest{};
    streaming_result_ = runtime::TaskResult{};
    streaming_audio_ = runtime::AudioBuffer{};
    streaming_audio_offset_values_ = 0;
    streaming_text_.clear();
    streaming_published_bytes_ = 0;
    streaming_windows_processed_ = 0;
    stream_started_ = false;
}

runtime::StreamEvent WhisperASRSession::process_audio_chunk(const runtime::AudioChunk & chunk) {
    require_prepared("Whisper process_audio_chunk()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Whisper process_audio_chunk() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Whisper process_audio_chunk() requires start_stream");
    }
    if (chunk.channels <= 0 || chunk.samples.size() % static_cast<size_t>(chunk.channels) != 0) {
        throw std::runtime_error("Whisper streaming audio chunk has invalid channel layout");
    }
    // Whisper 前端需 16k mono：先把 chunk 重采样成 16k mon, 再累积进滑窗缓冲。
    constexpr int kSampleRate = 16000;
    const auto mono = audio::convert_interleaved_audio_to_mono_linear_resampled(
        chunk.samples,
        chunk.sample_rate,
        chunk.channels,
        kSampleRate);
    runtime::AudioBuffer mono_buf;
    mono_buf.sample_rate = kSampleRate;
    mono_buf.channels = 1;
    mono_buf.samples = mono;
    if (streaming_audio_offset_values_ == streaming_audio_.samples.size() && streaming_audio_offset_values_ > 0) {
        streaming_audio_.samples.clear();
        streaming_audio_offset_values_ = 0;
    }
    runtime::append_audio_buffer(streaming_audio_, mono_buf);
    return process_available_stream_chunks(false);
}

runtime::TaskResult WhisperASRSession::finish_stream() {
    return finalize();
}

runtime::TaskResult WhisperASRSession::finalize() {
    require_prepared("Whisper finalize()");
    if (task_.mode != runtime::RunMode::Streaming) {
        throw std::runtime_error("Whisper finalize() requires a streaming session");
    }
    if (!stream_started_) {
        throw std::runtime_error("Whisper finalize() requires start_stream");
    }
    if (streaming_audio_offset_values_ > streaming_audio_.samples.size()) {
        throw std::runtime_error("Whisper streaming pending audio offset is out of range");
    }
    if (streaming_audio_offset_values_ == streaming_audio_.samples.size() && streaming_windows_processed_ == 0) {
        throw std::runtime_error("Whisper finalize() requires streamed audio");
    }
    (void) process_available_stream_chunks(true);
    if (!streaming_result_.text_output.has_value()) {
        streaming_result_.text_output =
            runtime::Transcript{"", streaming_request_.text_input.has_value() ? streaming_request_.text_input->language : ""};
    }
    stream_started_ = false;
    if (stream_event_sink_ != nullptr) {
        runtime::StreamEvent event;
        event.is_final = true;
        stream_event_sink_(event);
    }
    engine::debug::trace_log_scalar("whisper.session.stream.windows", streaming_windows_processed_);
    return streaming_result_;
}

runtime::StreamEvent WhisperASRSession::process_available_stream_chunks(bool final) {
    runtime::StreamEvent last_event;
    last_event.is_final = false;
    if (streaming_audio_.sample_rate <= 0 || streaming_audio_.channels <= 0) {
        return last_event;
    }
    if (streaming_audio_.channels != 1) {
        throw std::runtime_error("Whisper streaming requires mono 16k accumulative audio");
    }
    // 每窗口默认 5s（Whisper 30s 窗口实时性差）；可用 audio_chunk_seconds 覆盖。
    constexpr double kDefaultWindowSeconds = 5.0;
    constexpr int64_t kMaxWindowFrames = 480000;  // 30s @ 16k（whisper encoder 固定窗口上限）
    const auto seconds = engine::audio::parse_audio_chunk_seconds_override(streaming_request_.options)
        .value_or(kDefaultWindowSeconds);
    if (!(seconds > 0.0F)) {
        throw std::runtime_error("Whisper streaming audio_chunk_seconds must be positive");
    }
    const int64_t window_frames = static_cast<int64_t>(std::llround(
        static_cast<double>(seconds) * static_cast<double>(streaming_audio_.sample_rate)));
    if (window_frames <= 0) {
        throw std::runtime_error("Whisper streaming audio_chunk_seconds produced an empty chunk");
    }

    int64_t processed_chunks = 0;
    while (true) {
        const int64_t pending_frames = static_cast<int64_t>(
            streaming_audio_.samples.size() - streaming_audio_offset_values_);
        if (pending_frames <= 0 || (!final && pending_frames < window_frames)) {
            break;
        }
        const int64_t take_frames = std::min<int64_t>(
            final ? pending_frames : window_frames,
            kMaxWindowFrames);
        if (take_frames <= 0) {
            break;
        }
        runtime::AudioBuffer chunk;
        chunk.sample_rate = streaming_audio_.sample_rate;
        chunk.channels = 1;
        chunk.samples.assign(
            streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(streaming_audio_offset_values_),
            streaming_audio_.samples.begin() +
                static_cast<std::ptrdiff_t>(streaming_audio_offset_values_ + static_cast<size_t>(take_frames)));
        streaming_audio_offset_values_ += static_cast<size_t>(take_frames);
        last_event = process_one_stream_chunk(chunk);
        ++streaming_windows_processed_;
        ++processed_chunks;
        if (stream_event_sink_ != nullptr && last_event.partial_text.has_value()) {
            stream_event_sink_(last_event);
            last_event.partial_text.reset();
        }
    }
    if (processed_chunks > 0) {
        if (streaming_audio_offset_values_ == streaming_audio_.samples.size()) {
            streaming_audio_.samples.clear();
            streaming_audio_offset_values_ = 0;
        } else if (streaming_audio_offset_values_ > 1ull * 1024ull * 1024ull &&
                   streaming_audio_offset_values_ * 2 > streaming_audio_.samples.size()) {
            streaming_audio_.samples.erase(
                streaming_audio_.samples.begin(),
                streaming_audio_.samples.begin() + static_cast<std::ptrdiff_t>(streaming_audio_offset_values_));
            streaming_audio_offset_values_ = 0;
        }
    }
    return last_event;
}

runtime::StreamEvent WhisperASRSession::process_one_stream_chunk(const runtime::AudioBuffer & audio) {
    // 与 offline run() 相同的选项解析（应先归一化 whisper. 前缀 → 裸名），对单个 ≤30s 窗口做一次离线转写。
    const auto options = normalize_request_options(streaming_request_.options);
    std::string language = language_from_options(options);
    if (language == "auto") {
        language = runtime_.detect_language(audio);
    }
    engine::models::whisper::WhisperDecodeOptions opts;
    opts.language = language;
    opts.max_len = max_len_from_options(options);
    opts.translate = translate_from_options(options);
    opts.timestamps = timestamps_from_options(options);
    opts.beam_size = beam_size_from_options(options);
    opts.repetition_penalty = repetition_penalty_from_options(options);
    opts.clip_offset_seconds = clip_offset_from_options(options);
    const std::string output_language = opts.translate ? "en" : language;
    const auto transcription = runtime_.transcribe(audio, opts);

    runtime::StreamEvent event;
    event.is_final = false;
    if (transcription.text.empty()) {
        return event;
    }
    const std::string delta = streaming_text_.empty()
        ? transcription.text
        : " " + transcription.text;
    streaming_text_ += delta;
    if (!streaming_result_.text_output.has_value()) {
        streaming_result_.text_output = runtime::Transcript{"", output_language};
    } else if (streaming_result_.text_output->language.empty()) {
        streaming_result_.text_output->language = output_language;
    }
    streaming_result_.text_output->text = streaming_text_;
    if (streaming_published_bytes_ < streaming_text_.size()) {
        event.partial_text = runtime::Transcript{
            streaming_text_.substr(streaming_published_bytes_),
            streaming_result_.text_output->language,
        };
        streaming_published_bytes_ = streaming_text_.size();
    }
    return event;
}

runtime::IOfflineVoiceTaskSession & WhisperASRSession::vad_session() {
    // 懒加载内部 silero VAD：首次长音频分段（audio_chunk_mode=vad）时才加载权重。
    if (vad_session_ == nullptr) {
        runtime::ModelLoadRequest load_request;
        load_request.model_path = vad_model_path_;
        vad_model_ = engine::models::silero_vad::load_silero_vad_model(load_request);
        auto session = vad_model_->create_task_session(
            runtime::TaskSpec{runtime::VoiceTaskKind::Vad, runtime::RunMode::Offline},
            runtime::SessionOptions{options().backend, {}});
        auto * offline = dynamic_cast<runtime::IOfflineVoiceTaskSession *>(session.get());
        if (offline == nullptr) {
            throw std::runtime_error("Whisper internal VAD session does not support offline execution");
        }
        session.release();
        vad_session_.reset(offline);
    }
    return *vad_session_;
}

WhisperASRLoadedModel::WhisperASRLoadedModel(
    runtime::ModelMetadata metadata,
    runtime::CapabilitySet capabilities,
    std::shared_ptr<const WhisperWeights> weights)
    : metadata_(std::move(metadata)),
      capabilities_(std::move(capabilities)),
      weights_(std::move(weights)) {}

const runtime::ModelMetadata & WhisperASRLoadedModel::metadata() const noexcept {
    return metadata_;
}

const runtime::CapabilitySet & WhisperASRLoadedModel::capabilities() const noexcept {
    return capabilities_;
}

std::unique_ptr<runtime::IVoiceTaskSession> WhisperASRLoadedModel::create_task_session(
    const runtime::TaskSpec & task,
    const runtime::SessionOptions & options) const {
    return std::make_unique<WhisperASRSession>(task, options, weights_);
}

std::shared_ptr<runtime::IVoiceModelLoader> make_whisper_loader() {
    return std::make_shared<WhisperASRLoader>();
}

}  // namespace engine::models::whisper
