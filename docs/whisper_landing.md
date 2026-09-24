# Whisper 家族落地：项目规范分析 + 对齐后的实现蓝图

> 目的：把已用探针在 `temp/whisper_probe/` 验证过的 whisper 端到端转写（log-mel → encoder → decoder → token）按 audio.cpp 项目既有规范集成进主工程。
> 本文是「先分析项目代码结构与文档规范 → 再按规范调整实现方式」的结论与可执行蓝图，供后续逐文件落地。
> 所有结论均来自对既有源码/文档的只读查证（附证据位置），不臆测。

---

## 1. 项目规范要求汇总（证据出处）

### 1.1 代码风格 / header
- **无 license/copyright header 要求**：`src/models/` 下 446 个 `.cpp` 全部不带 `Copyright/SPDX`（canary/citrinet/framework 均直接以 `#include` 开头，如 `src/models/canary_asr/runtime.cpp:1`）。已落地文件（tokenizer）不带 header **合规**。
- `.gitattributes` 仅约束 `*.bat/*.cmd` 用 CRLF、二进制扩展名；与模型源码无关。

### 1.2 CMake 模型注册
- 宏定义在 `CMakeLists.txt:412-425`：`audiocpp_add_model(<name> SOURCES INCLUDES LOADERS ALIASES DEPENDS)`，所有关键字为**多值参数**（每个值单独一行）。
- **LOADERS 传带完整命名空间的 `make_<family>_loader` 函数名**；INCLUDES 传公开接口头；ALIASES 可多值（`CMakeLists.txt:649-662` roformer 两个 loader + 两个 alias 为样板）。
- 最简 offline ASR 范本：`CMakeLists.txt:1633-1642` citrinet_asr（3 个 .cpp、1 include、1 loader）。

### 1.3 家族文件布局
- 标准结构（以 `src/models/citrinet_asr/` 为最简 offline ASR 范本）：
  - `src/models/<family>/`：`assets.cpp`（权重+配置加载）、`runtime.cpp`（图构建+前向）、`session.cpp`（loader+session 契约）；权重亦可拆 `weights.cpp`（moonshine_asr）。
  - `include/engine/models/<family>/`：`assets.h`、`runtime.h`、`session.h`。
  - 公开接口头命名不强制：citrinet/moonshine 用 `session.h`，canary 用 `model.h`，均可。
  - 命名空间：`engine::models::<family>`。
- 现 whisper 家族 `src/models/whisper/` 已建，含 `tokenizer.cpp`；`include/engine/models/whisper/tokenizer.h` 已建。

### 1.4 Loader / Session 契约（规范必守）
参照 `citrinet_asr/session.cpp:34-191` 完整链路：
1. `class <X>Loader final : public runtime::IVoiceModelLoader`（`family()`、`advertised_capabilities()`、`can_load()`、`inspect()`、`load()`）。
2. `<X>LoadedModel final : public runtime::ILoadedVoiceModel`（`metadata()`、`capabilities()`、`create_task_session()`）。
3. `<X>Session final : public runtime::RuntimeSessionBase, public runtime::IOfflineVoiceTaskSession`（`family()`、`task_kind()`、`run_mode()`、`prepare()`、`run()`）。
4. `make_<family>_loader()` 返回 `std::shared_ptr<runtime::IVoiceModelLoader>`。
- Whisper 离线 ASR 能力声明：`capabilities.supported_tasks = {{runtime::VoiceTaskKind::Asr, {runtime::RunMode::Offline}}}`。
- 权重加载采用 `load_<family>_weights_cached` 缓存模式（`citrinet_asr/assets.cpp:223-240`）：`static std::mutex` + `static unordered_map<string, weak_ptr>` + `checkpoint_cache_key(require_file("weights"))`。
- 资源取用：`resources.open_tensor_source("weights")` 取权重；`resources.require_file("config")` / `require_file("tokenizer")` 取非权重资源。

### 1.5 Model Spec v1
- 版本常量 `kModelSpecSchemaVersion = 1`（`include/engine/framework/model_spec/schema.h:9`）；校验在 `src/framework/model_spec/schema.cpp:692-736`。
- `model_specs/*.json` 由 `CMakeLists.txt:23` 自动 GLOB，**无需手动登记**。
- 必填顶层键：`family, display_name, category, status, tasks, modes, languages, runtime, capabilities, options, packages, dependencies, ui, sources`（`dependencies` 必须为数组，空也要 `[]`；`options` 必填）。
- **权重 vs 非权重资源区分**：
  - 权重（safetensors/gguf）→ `sources[].tensors`（loader 端 `open_tensor_source("weights")`）。
  - 非权重资源（`config.json`、`multilingual.tiktoken`）→ `sources[].files` 或 `optional_files`（loader 端 `require_file("xxx")`）。
  - 校验只允许 `roots/files/optional_files/tensors/optional_tensors`（`schema.cpp:486-512`）。

### 1.6 Tokenizer 规范
- 项目无 `.tiktoken` 加载类，但 `HuggingFaceTokenizerJson`（`include/engine/framework/tokenizers/hf_tokenizer_json.h`）已被生产使用：`parakeet_tdt`、`nemotron_asr`、`granite5asr` 均经 `load_huggingface_tokenizer_json` 加载；`granite5asr` 明确为 "ByteLevel BPE"。
- 已落地 whisper tokenizer 用 `HuggingFaceTokenizerJson(id_to_token, "", true, true)`（byte_level GPT-2 解码）**与规范同类，合规**。
- 公开 API：`decode_ids(const std::vector<int32_t>&)`，可选 `skip_token_id`。

### 1.7 框架模块改动规则（CONTRIBUTING:27-29, 83-87）
- **Additive 优先**，勿改共享行为；变体应做独立实验模块（`xxxExp`），全仓当前无 `*Exp` 先例。
- **head=12 bug 影响面**：`whisper_frontend.cpp:132` `infer_hf_encoder_config` 硬编码 `n_audio_head=12`，仅在 `load_hf_encoder_layout`（`:329`）内调用；**全项目仅 `seed_vc` 走该路径受影响**（`moss_transcribe_diarize`/`vevo2` 走 `load_openai_layout` 不受影响）。
- **结论**：whisper 家族走 `load_openai_layout`（外部传入正确 `WhisperEmbeddingConfig`，head=6）或自行加载 encoder，**不改共享前端**（不影响 seed_vc）。

### 1.8 文档同步清单（新增家族须补）
- `README.md:106-129`「Speech Recognition And Analysis」表加 whisper 行（当前无）。
- `docs/asr.md:3-23` 模型表加行。
- 可选 `docs/models/whisper.md`（仿 `fun_asr_nano.md` 骨架）。
- `model_manager` catalog 编译进 exe（`app/model_manager/main.cpp:32`），无需独立维护。

---

## 2. 已落地部分（tokenizer）合规性对照

| 项 | 现状 | 是否合规 |
|---|---|---|
| 位置 | `src/models/whisper/tokenizer.cpp` + `include/engine/models/whisper/tokenizer.h` | ✅ 家族目录与命名空间 `engine::models::whisper` |
| license header | 无 | ✅ 与全仓一致 |
| 解码路径 | `HuggingFaceTokenizerJson(..., "", true, true)` byte_level | ✅ 与 graniteByteLevel 同类、parakeet/nemotron 同 API 族 |
| 依赖 | 复用框架 `hf_tokenizer_json`，无新增第三方 | ✅ |

> 说明：`tokenizer.cpp` 当前尚未列入任何 CMake target（`src/models/whisper/` 只有它一个文件且无 `audiocpp_add_model` 块），因此**尚未被编译进主工程**。需随完整模型块一起注册。

---

## 3. 关键技术决策（规范约束 + 实测证据）

### 3.1 Encoder：走 `load_openai_layout`，规避 head=12 共享 bug
- 用 `WhisperFrontendComponent::load_openai_layout(source, backend, config)` 并显式传 `WhisperEmbeddingConfig{ n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4 }`（whisper-tiny 实测值，A2 已对拍 RMSE 3e-6）。
- 编码可用 `encode_log_mel()`（`whisper_frontend.h:52`）或 `WhisperEmbeddingModule::build`（行为等价，canary 构图为模板）。
- **不改** `whisper_frontend.cpp` 的 `n_audio_head=12`（CONTRIBUTING additive；且仅 seed_vc 受影响，非本家族职责）。

### 3.2 Decoder：采用「全序列 build」路径（本轮实测裁决）
- 设计文档 §4.2 原拟 `TransformerDecoderBlockModule::build_cached_tail`（Canary 正统流式）。但 **A3→M1 实测**揭示：
  - 全序列 `build` 路径：真实 whisper-tiny 权重 hidden RMSE 8.3e-6、logits 6.7e-6、argmax 8/8；本轮 full-seq 转写探针输出 `[542,8710,60]` → tokenizer → `"[Music]"`，与 numpy 参照**逐 token + top3 全等**。
  - `build_cached_tail` 增长 cached KV（>2 key）时 flash 分叉（pos=4 起值不一致，非 NaN）。
- **裁决**：M1 正式 runtime decoder 首版用**全序列 build**（保证正确、可复现 `[Music]`）；KV-cache 流式留作后续独立优化（届时按设计文档 §4.2 重建正统 Kaiser 图并专项 debug flash）。
- decoder 权重：`model.decoder.embed_tokens/embed_positions/layer_norm/layers.*`（HF 命名，探针已实锤）；`decoder_attention_heads=6`；tied lm_head（复用 `embed_tokens` 转置）。

### 3.3 权重来源
- 项目 spec/资源包约定支持 safetensors；whisper 家族沿用 HF safetensors 权重（`model.safetensors`）+ `config.json` + `multilingual.tiktoken` 三件套（此三件已在 `temp/whisper_probe/whisper-tiny/` 下载并验证）。

---

## 4. 落地蓝图（逐文件，按规范实施）

> 全部以 `citrinet_asr` 为结构模板、`canary_asr/runtime.cpp` 为图构建模板；逻辑已在探针验证，`src/models/whisper/` 目录与 tokenizer 已就位。

### 4.1 新增文件
1. **`include/engine/models/whisper/assets.h`** — `WhisperConfig`（d_model/encoder·decoder heads/layers/vocab/mels/token ids）、`WhisperWeights`（encoder 前端 + decoder 各层权重 + tokenizer，持 `shared_ptr<const TensorSource>`）、`load_whisper_weights_cached(model_path)`。
2. **`src/models/whisper/assets.cpp`** — `config.json` 解析（d_model、encoder/decoder_attention_heads、num_*_layers、vocab_size、num_mel_bins）、`resources.open_tensor_source("weights")` + `resources.require_file("config")`/`require_file("tokenizer")`、`HuggingFaceTokenizerJson` 构建、`load_whisper_weights_cached`（mutex+weak_ptr cache）。
3. **`include/engine/models/whisper/runtime.h`** — `WhisperRuntime`（持 `WhisperFrontendComponent` encoder + decoder 图 + `WhisperTokenizer`），`transcribe_audio(audio)->Transcript`；含 `WhisperTranscriptionResult{ text, token_ids, logits }`。
4. **`src/models/whisper/runtime.cpp`** — 构图：encoder 参考 canary `build_encoder`（`WhisperEmbeddingModule::build` + 全序列 decoder build，非 build_cached_tail）；`run()` 支持可变前缀 L（探针 `probe_whisper_transcribe_full.cpp` 逻辑）；贪心解码 `[sot, lang, transcribe, notimestamps]` 起。
5. **`include/engine/models/whisper/session.h`** + **`src/models/whisper/session.cpp`** — citrinet 全链路照搬：`WhisperASRLoader`/`WhisperASRLoadedModel`/`WhisperASRSession` + `make_whisper_loader()`；`prepare()` 校验音频、`run()` 走 `runtime_.transcribe_audio` 输出 `Transcript`。

### 4.2 `CMakeLists.txt` 注册块（紧邻其它 ASR 家族）
```cmake
audiocpp_add_model(whisper
    SOURCES
        src/models/whisper/assets.cpp
        src/models/whisper/runtime.cpp
        src/models/whisper/session.cpp
        src/models/whisper/tokenizer.cpp
    INCLUDES
        engine/models/whisper/session.h
    LOADERS
        engine::models::whisper::make_whisper_loader
)
```

### 4.3 `model_specs/whisper.json`（schema v1）
- 顶层：`schema_version:1, family:"whisper", category:"asr", status:"wip", tasks:["asr"], modes:["offline"], languages:["auto","en","ja",...], dependencies:[]`，`options:{request/session/load}`，`ui:{docs:["docs/asr.md","docs/gguf.md"]}`。
- `sources[0]`：`roots.model="."`；`files:{"config":"model:config.json","tokenizer":"model:multilingual.tiktoken"}`（非权重）；`tensors:{"weights":"model:model.safetensors"}`（权重）。格式见 `model_specs/fun_asr_nano.json` 同款。

### 4.4 文档同步
- `README.md` ASR 表 + `docs/asr.md` 模型表各加 whisper 行；新增 `docs/models/whisper.md`（仿 `fun_asr_nano.md` 骨架：Install/CLI/Options/GGUF/Validation）。

---

## 5. 执行顺序与验证
1. 写 `assets.{h,cpp}` → `runtime.{h,cpp}` → `session.{h,cpp}`（复用探针已验证逻辑，非重写数学）。
2. CMake 注册 + `model_specs/whisper.json` + 文档同步。
3. 构建 `engine`（或 `audiocpp_cli`），确认 whisper target 编译链接通过。
4. 以 `temp/whisper_probe/whisper-tiny/` 作为模型目录跑 `audiocpp_cli --task asr --family whisper --model <dir> --audio librispeech_0000.wav`，断言输出 `[Music]`（与 numpy/whisper 真值一致，复现探针结果）。

> 注：`[Music]` 为 librispeech 0000 片段的真实转写（其余探针 clip 可逐步补真值对拍）。

## 6. 未尽事项 / 后续
> 蓝图落地后的状态（2026-09-20）：以下原「未尽项」多数已完成并在
> `docs/models/whisper.md` 实证，标注为 ✅；仅 seed_vc 相关仍在 whisper 家族
> 之外。
- ✅ **KV-cache 流式**：已完成单 batch 单步累积 KV 解码（`TransformerKVCache` +
  逐 token 前向），三路径实证 `[Music]` 回归。
- ✅ **GGUF 权重包**：已完成（`audiocpp_gguf` 打包 f16/q8_0 GGUF，见
  `docs/models/whisper.md`「Build A GGUF」）。
- ✅ **长音频分段 / 时间戳 / translate / auto 语言检测 / Silero VAD / beam
  search / word_timestamps**：均已落地并 CLI+v2 端点实证（详见
  `docs/models/whisper.md`）。
- ✅ **CUDA 后端串联**：已完成（2026-09-20）。`--backend cuda` 下
  KV-cache 流式 / beam / timestamps / translate / 长音频分段全部与 CPU 逐
  token 全等；327s 长音频贪心 RTF=0.0018（较 CPU 7.6×、较同规模
  faster-whisper 6.8×）。详见 `docs/whisper_cuda_plan.md` §7.5。
- **head=12 共享 bug**：仅 seed_vc 受影响，属 seed_vc 家族职责；建议后续为
  seed_vc 做独立 `whisper_frontend_exp` 变体（CONTRIBUTING xxxExp），不在
  whisper 家族内处理。
- **多语质量**：whisper-tiny 对所有非英语语言质量偏低（translate 固定输出
  英文；日→中等跨语翻译不在标准能力内），文档「Known limitations」已如实
  记录，非实现缺陷。
