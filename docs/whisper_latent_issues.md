# Whisper 模型：其余类似隐患审计清单

> 目的：承 `docs/whisper_beam_optimization.md` §8 发现的 `dec_cross_` F32 精度隐患，系统排查同性质的「该用半精度没用 / 机制上隐藏的显存与 RAM 浪费」隐患。
> 方法：以 whisper 仓库 own 代码（`src/models/whisper/runtime.cpp` + 框架 `kv_cache`）逐 buffer 审计，并对照 faster-whisper/框架其他模型的良好实践。
> 证据标注：⚠️=已验证源码 / 🔶=推导（未实测，需回填）。数值均为 tiny / large-v3 两种量级（tiny: d384/L4/H6；large: d1280/L32/H20）。

---

## 0. 隐患速览（按优先级）

| # | 隐患 | 类别 | tiny 收益 | large 收益 | 改动量 | 证据 |
|---|---|---|---|---|---|---|
| I1 | `dec_cross_` 写死 F32（已§8 报） | 显存 | −8.8 MB | **−245.8 MB/份** | ✅已落地 | ⚠️ |
| I2 | encoder memory（`kv_memory_`/`encoder_output_`）F32 常驻 | 显存 | −1.1 MB | −7.3 MB（×1份） | ✅已随I1打通 | ⚠️ |
| I3 | **beam 的 CPU KV 快照 `TransformerKVState` F32** | RAM+性能 | +89.6 MB/候 | **+546 MB/候,×4≈2.1 GB** | 大(§6) | ⚠️ |
| I4 | `TransformerKVCache` 每层 2×F32 CPU scratch 常驻 | RAM | ~2.8 MB | **~140 MB** | 小 | ⚠️ |
| I5 | 贪心/beam 每步**全量读回 dec_logits_**（51865 F32→CPU 采样） | 性能 | 每步 ~200 KB D2H | 同 | 小 | ⚠️ |
| I6 | 贪心路径 KV 常驻但 **cross K/V 未复用**：贪心每步 decoder 图重算/读取 F32 cross | 性能 | 低 | 中 | 小 | 🔶 |
| I7 | 权重/计算精度已达标（对照组）——**非隐患**，仅记录 | — | none | none | — | ⚠️ |
| I8 | `kWeightContextBytes=1GB`/`kGraphContextBytes=256MB` 巨型 context 上限 | 风险 | 预留大 | 预留大 | 核查即可 | ⚠️ |

> 排序逻辑：I1（半精度没用的量级点）→ I2（同性质上游）→ I3（§6 已知的 beam CPU 快照）→ I4（框架每层 scratch）→ I5/I6（性能）→ I8（预留 review）。

---

## 1. I1：dec_cross_ 写死 F32（§8 已报，复核）

- 位置：`runtime.cpp:620/622`，`GGML_TYPE_F32`；decoder cross-attention 的目标 KV buffer。
- 现状：GPU 上每层 cross K/V = `[1, heads, mem_frames, head_dim]` **F32**（mem_frames=max_source=1500）。
- 收益：large 下 491.5→245.8 MB/份（−245.8），batch=4 时再放大。
- 处置：**✅ 已实现（2026-09-20）**——不是写死 F16（会重蹈 F32 覆辙），而是做成**运行期可配精度选项**：
  - `WhisperKVStoragePrecision`（F32 默认 / F16 / BF16）+ `WhisperRuntimeOptions.cross_kv_precision`，`WhisperRuntime` 构造第三参（带默认值，旧调用零改动）；
  - `build_kv_graphs` 按选项选 `cross_type`，`dec_cross_` 两处 make_tensor 用 `cross_type` 而非硬编码 F32；
  - 刻意不塞进 `WhisperConfig`（config.json 的模型元数据，放运行期选项语义更对）。语义对齐 faster-whisper `--compute_type`。
- 安全性已验证：cross K/V 仅在 cross_kv 图 `ggml_cpy`（F32→目标类型自动转）+ decoder cross-attn matmul 消费，无 CPU 直读；beam 的 export/import 只走 self-KV（`kv_cache_`）不受影响；`validate_cross_cache` 仅查 shape 不查 type → F16/BF16 输入标准支持。**engine_model_whisper 编译通过。**
- 用法：`WhisperRuntime(w, exec, {.cross_kv_precision=WhisperKVStoragePrecision::F16})` → large 下 `dec_cross_` 显存 **−245.8 MB/份**。

## 2. I2：encoder memory 链 F32 常驻（同 I1 性质，上游）

- 位置：`runtime.cpp:605` `kv_memory_` F32、`encoder_output_`（575 `ggml_set_output`，由 allocator 常驻）F32。
- 现状：encoder 输出 melody → cross K/V 投影的输入是 F32；whisper 官方在 GPU 上 encoder 输出常为 f16。
- 收益：memory 从 F32 降 F16，large 下 7.3→3.7 MB（单份，小）；但它是 cross K/V（I1）的上游，**一起降才能把 cross 彻底半精度化**，且省 encoder→decoder 的转换。
- 注意：encoder 内部 conv/GELU 链是 F32（ggml 默认中间精度），只降 memory 端点数有限；主要价值在打通 I1。

## 3. I3：beam 的 CPU KV 快照 F32（§6 已报，RAM 大头）

- 位置：`runtime.cpp:818` `BeamCandidate.kv = TransformerKVState`（`kv_cache.h:12-20` 用 `std::vector<float>`）；每候选全量 KV 走 CPU。
- 现状：beam=4 时 large 下 **~2.1 GB CPU RAM**（每候选 ~546 MB F32），且每步 import/export 全量 D2H/H2D。
- 收益：§6 方案（GPU 内 constant batch）移除 CPU 快照 → RAM **−1.6~2.1 GB** + beam RTF 0.232→~0.03。
- 处置：依赖 §6 的大改造（框架 batch 基座 + whisper 图 batch 化）。**若不 batch，至少应把快照存 F16（每候选 RAM 减半，~1.05 GB），零机制改动。**

## 4. I4：TransformerKVCache 每层 2×F32 CPU scratch 常驻

- 位置：`kv_cache.cpp:119-120`，`LayerCache.import_key_scratch/import_value_scratch = std::vector<float>(cache_elems)`。
- 现状：每个 layer 预分配 2 个 F32 CPU 缓冲（用于 import_state 的暂存），whisper decoder_L 层全部常驻。
- 收益：large：每层 448×1280×4×2=4.4 MB，×32 层 = **140 MB**（仅 whisper 一个模型实例）；tiny ~2.8 MB。
- 处置：scratch 可随用随分配 / 用 F16 暂存 / 只在需要 import 时分配——但它是框架通用 KV cache，改动涉及其他模型（qwen/yue2）行为，需谨慎。**列为低频优化或框架统一优化项。**

## 5. I5：每步全量读回 dec_logits_（采样在 CPU）

- 位置：`runtime.cpp:759`（贪心）`read_tensor_f32_into(dec_logits_)`；beam `850` 同理。
- 现状：每次 token 生成都把整行 `[1, vocab=51865]` F32（~200 KB）从 GPU 拷回 CPU 做 argmax/top-k。对贪心只需 argmax，但整行读。
- 收益：性能微小（200 KB D2H/步，long 音频几百步 ≈ 几十 MB 传输+延迟）；同词表下 all 模型都这么干（框架惯例 60+ 模型零例外），**不属异常，仅记录**。

## 6. I6：贪心路径 cross K/V 每步重读 F32（推导）

- 位置：`build_cross_kv_graph`（636-651）encoder 后投影一次进 `dec_cross_` F32；贪心每步 `build_cached_tail` 读取。
- 现状：cross K/V 在 GPU 上是 F32 每步被 decoder 消费；若 I1 降 F16，每步读取带宽减半。
- 收益：与 I1 合并，不独立计。

## 7. I7：权重/计算精度——对照组（非隐患）

- `load_tensor(..., st, ...)` 按 GGUF 存储类型加载（283 行注释明确"避免强制 F32 吃掉量化红利"），bias 例外用 F32（可接受）。
- 结论：**权重侧已达标**，不存在像 dec_cross_ 那样的整权重 F32 浪费。这是"已经做对"的参照，反向证明 I1/I2 是遗漏而非普遍疏忽。

## 8. I8：巨型 context 上限（风险，需 review）

- `kWeightContextBytes=1GB`（47）、`kGraphContextBytes=256MB`（48），用于 encoder/decoder/cross_kv 三个 `ggml_init`。
- 现状：这些是 context **预留上限**，不是实际分配（ggml context 按需增长）；但 1GB 用于"权重"context 若绑定了 backend buffer 会抬高显存预算。
- 处置：核实 backend weight store 是否按此上限一次性向 GPU 索要；若是，改成按实际权重大小而非 1GB 上限。

---

## 9. 汇总：优先做什么

| 优先级 | 动作 | 依赖 | 收益(large) |
|---|---|---|---|
| P0 | I1 把 `dec_cross_` 精度**做成可配项**（F32 默认 / F16 / BF16） | ✅已落地+编译通过 | −245.8 MB/份 显存 |
| P0 | I3-mini 把 beam CPU 快照存 F16 | 无 | −1.05 GB RAM |
| P1 | I2 memory 链随 I1 打通 | I1 | −7.3 MB + 去掉转换 |
| P2 | I4 KV cache scratch 用 F16/按需 | 框架级(影响他模型) | −140 MB RAM |
| P2 | I8 context 上限 review | 核查 | 视 backend 而定 |
| — | I5 不必动（框架惯例） | — | — |

> 与 faster-whisper 对照：它"该省的全省了"（bf16 贯穿、无 CPU 快照、KV 常驻 GPU、短句短 KV）；我们当前**显存侧最大的白捡是 I1（cross 精度）**，**RAM 侧最大的是 I3（beam CPU 快照）**；两者都独立于"是否做 batch 大改"，可先低风险落地。

---

*版本 v1 / 2026-09-20 22:5x。基于 runtime.cpp + kv_cache.h/.cpp 源码审计；⚠️ 纯源码确认，🔶 为推导需实测回填。*
