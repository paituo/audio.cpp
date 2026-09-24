#pragma once

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session_base.h"
#include "engine/models/whisper/assets.h"
#include "engine/models/whisper/runtime.h"

#include <filesystem>
#include <memory>

namespace engine::runtime {
class ILoadedVoiceModel;
class IOfflineVoiceTaskSession;
}

namespace engine::models::whisper {

class WhisperASRSession final
    : public runtime::RuntimeSessionBase
    , public runtime::IOfflineVoiceTaskSession
    , public runtime::IStreamingVoiceTaskSession {
public:
    WhisperASRSession(
        runtime::TaskSpec task,
        runtime::SessionOptions options,
        std::shared_ptr<const WhisperWeights> weights);
    ~WhisperASRSession() override;

    std::string family() const override;
    runtime::VoiceTaskKind task_kind() const override;
    runtime::RunMode run_mode() const override;
    void prepare(const runtime::SessionPreparationRequest & request) override;
    runtime::TaskResult run(const runtime::TaskRequest & request) override;

    // streaming session（方案 A：滑窗累积音频 → 复用 offline transcribe → 增量 partial text）。
    runtime::StreamingPolicy streaming_policy() const override;
    void start_stream(const runtime::TaskRequest & request) override;
    void set_stream_event_sink(runtime::StreamEventCallback sink) override;
    void reset() override;
    runtime::StreamEvent process_audio_chunk(const runtime::AudioChunk & chunk) override;
    runtime::TaskResult finish_stream() override;
    runtime::TaskResult finalize() override;

private:
    // 懒加载内部 VAD session 的访问器：供长音频分段（AudioChunkMode::Vad）复用 silero_vad。
    runtime::IOfflineVoiceTaskSession & vad_session();

    // streaming：把当前累积的 16k mono 滑窗按 audio_chunk_seconds 切段，逐段转写并输出增量。
    runtime::StreamEvent process_available_stream_chunks(bool final);
    runtime::StreamEvent process_one_stream_chunk(const runtime::AudioBuffer & audio);

    runtime::TaskSpec task_;
    WhisperRuntime runtime_;
    std::filesystem::path vad_model_path_;
    std::unique_ptr<runtime::ILoadedVoiceModel> vad_model_;
    std::unique_ptr<runtime::IOfflineVoiceTaskSession> vad_session_;

    // streaming 状态。
    runtime::TaskRequest streaming_request_;
    runtime::TaskResult streaming_result_;
    runtime::AudioBuffer streaming_audio_;      // 已重采样为 16k mono 的累积音频
    size_t streaming_audio_offset_values_ = 0;  // 已消费样本（values 计数）
    std::string streaming_text_;                // 全量累积文本
    size_t streaming_published_bytes_ = 0;      // 已通过 partial_text 发布过的字节数
    int64_t streaming_windows_processed_ = 0;
    runtime::StreamEventCallback stream_event_sink_;
    bool stream_started_ = false;
};

class WhisperASRLoadedModel final : public runtime::ILoadedVoiceModel {
public:
    WhisperASRLoadedModel(
        runtime::ModelMetadata metadata,
        runtime::CapabilitySet capabilities,
        std::shared_ptr<const WhisperWeights> weights);

    const runtime::ModelMetadata & metadata() const noexcept override;
    const runtime::CapabilitySet & capabilities() const noexcept override;
    std::unique_ptr<runtime::IVoiceTaskSession> create_task_session(
        const runtime::TaskSpec & task,
        const runtime::SessionOptions & options) const override;

private:
    runtime::ModelMetadata metadata_;
    runtime::CapabilitySet capabilities_;
    std::shared_ptr<const WhisperWeights> weights_;
};

std::shared_ptr<runtime::IVoiceModelLoader> make_whisper_loader();

}  // namespace engine::models::whisper
