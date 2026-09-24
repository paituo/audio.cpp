# Whisper Beam 性能攻坚方案（我们仓库内落地施工图）

> 目的：在 `src/models/whisper/runtime.cpp`（`--family whisper`）内，把 beam search 从「CPU KV 往返 + 逐候选串行」改造成「GPU 内恒定 batch」方案，对齐 faster-whisper（ctranslate2）的性能量级。
> 状态：**方案分析产物，未实施**。分级：A/B 框架基座已验证存在；whisper 侧 batch 化属源码可推导的增量改造，**最终达标须跑通后实测回填**。
> 归属：承接 `docs/whisper_cuda_plan.md` §7.6 的框架级分析，本节落到具体代码改动清单。

---

## 1. 现状与根因（代码定位）

`transcribe_beam_kv`（`runtime.cpp:822-923`）每一解码步，对每个活跃候选做：

```
for 每个候选 cand:
    kv_cache_.import_state(cand.kv)          # D2H+H2D? 否——CPU快照 import 回显存 (ggml_backend_tensor_set)
    forward_single_position(cand.tokens.back(), ...)  # 单 batch [1,1,D] 前向 (ggml_backend_graph_compute)
    kv_cache_.export_state() → cand.kv       # 整 KV 拉回 CPU std::vector<float>
    top_k_log_softmax(logits, beam_size)      # 全词表(51865) log-softmax + partial_sort
```

实测（327.6s 长音频，GPU RTX 5070 Ti）：greedy RTF 0.00715 → beam=1 RTF 0.01288（1.8×）→ beam=4 RTF 0.21555（**30×**）。faster-whisper beam=5 仅 0.0314。

三层放大（每步）：
1. **结构固有 1.8×**（beam=1 vs greedy）：beam 路径每步强制 `import→export` 全量 KV 往返 + 全词表 log-softmax/partial_sort + 候选 struct 拷贝；贪心 `advance_after_direct_append` 无往返、KV 常驻逐步 append。
2. **候选串行超线性 16.7×**（beam=4 vs beam=1）：4 个候选是 4 次**串行**单 batch 前向（每个 = kernel launch + 4 层各自全量 KV 往返），且 D2H/H2D 同步**阻塞 GPU**，算力吃不满。
3. cross-KV 虽共享一份（`dec_cross_`），但每候选前向时都要重新用它，不随 batch 放大（计算上共享）。

**根因一句话**：beam = 「每步 B 次串行单 batch 前向 + 每候选每次全量 KV 在显存↔CPU 往返」；faster-whisper = 「每步一次 batch=B 前向 + KV 全常驻显存」。

---

## 2. 目标形态

每解码步**只做一次 batch=B 的前向**，KV 全部常驻 GPU，采样回 CPU 打分重排（沿用框架「采样回 CPU」惯例，60+ 模型普查零例外）。

```
每步:
  把 B 个活跃候选的末位 token 打包成 dec_token_in_ = [B]
  → 一次 forward（decoder 图 batch=B, KV batch=B 常驻）
  → 读回 dec_logits_ = [B, vocab]（只读每候选一行）
  → 对每候选 top_k_log_softmax + 全局 top-beam_size 重排（现有逻辑不变）
  → 下一轮重填 batch（KV 在 batch 槽位间移动/清零，不出设备）
```

---

## 3. 代码改动清单

### 3.1 框架层：`TransformerDecoderBlockModule` 新增 batch tail build

现状：`transformer_blocks.h:260` 只有 `build_cached_tail`（单 batch 布局：输入 `x[1,1,D]`、KV `[1,kv_steps,H,d]`、单 slot）。

需新增（参照已有先例）：
- **`qwen_causal_decode_runtime.cpp`**：`build_causal_decode_batched` / `prefill_tokens_batched` / `write_batched_cached_step_mask` —— batch KV 游标 + batch mask + batch 前向的完整模板。
- **`yue2/ar_runtime.cpp` + `minimax_music3/ar_runtime.cpp`**：`TransformerBatchedKVState`（`valid_steps_by_batch` / `current_end_by_batch`，**每 batch 行独立步数游标**）—— 这正好匹配 beam 语义（各候选前缀长度可不同）。

新增 `build_cached_tail_batched`：token/pos 输入 `[B,1]`、KV `[B,kv_steps,H,d]`、slot `[B]`、causal mask `[B,kv_steps]`，内部 self-attn 与 FFN 复用现有算子（ggml 算子本身 batch 无关）。

### 3.2 whisper 模型层：`runtime.cpp`

**3.2.1 decoder 图 batch 化**（`build_decoder_kv_graph`，654-691）
- `dec_token_in_`/`dec_pos_in_`：`[1,1]` → `[B,1]`（I32）
- `dec_slot_`：`[1]` → `[B]`
- `dec_causal_mask_`：`[1,kv_steps]` → `[B,kv_steps]`；写掩码时每 batch 行放行自己的 `current_end`
- `dec_keys_/dec_values_`：`[1,kv_steps,H,d]` → `[B,kv_steps,H,d]`
- `dec_logits_`：`[1,vocab]` → `[B,vocab]`

**3.2.2 KV cache 换 batch 基座**
- `kv_cache_`：`TransformerKVCache` → `TransformerBatchedKVCache`（`kv_cache.h:105-151`，框架已有）
- beam 候选不再持有 CPU `BeamCandidate.kv`（`runtime.cpp:818`），改为「batch 槽位号」；候选增删时在 batch 槽位间 `ggml_backend_tensor_set` 行级移动/清零（显存内重排，非跨设备）

**3.2.3 cross-attention batch 对齐（★ 关键修正点，§7.6.2 需修正）**
- whisper cross-attn 模块 `validate_cross_cache`（`cross_attention.cpp:53-55`）**强制 `key_value.key.shape.dims[0] == query.shape.dims[0]`**，不支持「query batch=B、cross K/V batch=1」的自动广播。
- 因此 batch 图里 cross K/V 必须是 `[B, heads, mem_frames, head_dim]`。
- 实现：encoder memory 的 K/V 投影**只做一次**（现有 `build_cross_kv_graph` 产出 `[1,H,mem,d]`），在 decoder 图里用 `ggml_repeat` 复制成 B 份（或共享 tensor + 手工广播）。
- 成本量化（whisper-tiny，mem_frames=max_source_positions=1500，H=6，d=64）：单层 cross KV = 1×6×1500×64×2 ≈ 4.6MB；B=4 → 复制 4 份 ≈ 18MB，**encoder 预算时复制一次**（非每步），可接受。大模型+大 beam 时线性放大，需评估但 ASR（beam≤8）可承受。
- ⚠️ §7.6.2 写「cross 免费共享（batch=1）」**过于乐观**：计算上共享是对的，但张量布局需 B 份（batch 对齐）。本施工图以本条目为准。

**3.2.4 beam 主循环 batch 化**（`transcribe_beam_kv`，822-923）
- 把"逐候选 forward_single_position"（845-874）改为：先收集所有候选当前 logits 行 → 打包 `[B]` token → 一次 batch 前向 → 读回 `[B,vocab]`
- top-k 打分 / 全局 top-beam_size 排序 / `pick_score` 最终选择（879-922）**逻辑保持不变**（这些是算法层，与 batch 无关）
- 每步更新 batch 槽位的 KV（`valid_steps_by_batch`）

### 3.3 贪心路径不动
`transcribe_greedy_kv`（723-778）已是最优（无往返、KV 常驻 append），保持不变。可用 `TransformerBatchedKVCache` 但 batch=1 与现贪心等价的简化路径。

---

## 4. 正确性与回归验证

| 验证项 | 方法 |
|---|---|
| 贪心不变 | 贪心走原图，回归现有 greedy 测试（CPU/CUDA 逐 token 全等，已有基线） |
| beam 算法正确 | CPU/CUDA 下，旧 beam(往返) vs 新 batch beam **逐 token 全等**（beam_size=2/4/5 + timestamps） |
| 跨候选独立 | 各候选 `valid_steps_by_batch` 独立性断言（长度不同也能打包 batch） |
| 时间戳/translate | 复用 §7.5 六类路径回归 |
| 性能 | 327s 长音频 RTF 重测，目标 beam=4 从 0.232 → faster-whisper 量级（~0.03，~7.7×） |
| 内存/显存 | 复测 §7.6.3：期望 RAM 回落（不再 CPU KV 快照）、VRAM 小幅上升（batch KV + cross 复制） |

## 5. 工作量与风险

| 项 | 范围 | 工作量 | 风险 |
|---|---|---|---|
| 框架 `build_cached_tail_batched` | transformer_blocks.h/.cpp | **中-高（主要项）** | 需新增并实测 batch attention 正确性 |
| cross K/V broadcast(B 份) | runtime.cpp decoder 图 | 中 | ggml_repeat 布局需实测 |
| whisper 图 batch 化接线 | runtime.cpp | 中 | 输入/mask/slot 维对齐 |
| beam 循环 batch 化 | runtime.cpp | 中 | 复用现有 top-k/排序逻辑 |
| 回归 + 性能/内存复测 | 测试 | 中 | — |

> 结论：**框架 batch 基座（`TransformerBatchedKVCache` + qwen/yue2 batched decode）已验证存在**，whisper batch 化是**可推导的增量改造，主要工程量在「框架层新增 batch tail build」+「cross K/V 复制」两处**。若 beam 质量是硬需求（如 JAV 日→中长句翻译），值得投入；若贪心已满足业务，贪心已碾压 faster-whisper（快 6.8×）可暂缓。

---
*版本 v1 / 2026-09-20。承接 §7.6 框架级分析，落到 whisper 仓库内具体改动清单。§7.6.2「cross 免费共享」修正为「计算共享、布局需 B 份」（cross_attention.cpp:53-55 强制 batch 对齐）。*

---

## 6. large-v3 内存/显存变化评估 + 修改边界

### 6.1 large-v3 维度（本地 `models/whisper-large-v3/hf/config.json`）

| 参数 | large-v3 | (对比 whisper-tiny) |
|---|---|---|
| d_model | 1280 | (384) |
| decoder/encoder layers | 32 | (4) |
| decoder heads | 20 (head_dim=64) | (6) |
| max_source_positions (mem_frames) | 1500 | (1500) |
| max_target_positions (kv_steps) | 448 | (448) |
| vocab | 51866 | (51865) |

### 6.2 batch beam 对 large-v3 的内存/显存增量（beam=4，f16）

> ⚠️ **基数修正（v1.1，2026-09-20 22:3x）**：初版的「单候选 245.8 MB（f16）」**报错了当前 cross 的精度**——我们当前 `dec_cross_`（runtime.cpp:620-621）实际是 **F32（4 字节）＝ 491.5 MB/份**，不是 f16。f16 单份 245.76 MB 只是"batch 方案若改用半精度"的假设值。下表以**当前真实基数**重算，并区分「batch 方案沿用 F32」与「顺带降 F16」两种情形。

| 项 | 当前 GPU 常驻(1份,实际精度) | batch=4 沿用精度 | 净增 |
|---|---|---|---|
| decoder 自注意力 KV（32层满 448 步） | 35.0×2 = 70.0 MB（F16） | 280.0 MB | **+210.0 MB** |
| cross K/V（32层, 当前 **F32** 491.5MB/份） | 491.5 MB（1份） | **1966 MB（F32×4）** | +1474.5 MB |
| cross K/V（若 batch 时顺带降 **F16/bf16** 234.4MB/份） | 491.5 MB（1份） | 937.5 MB（F16×4） | **+446.0 MB** |
| dec_logits [B,vocab] F32 | — | 0.83 MB | ~0 |
| **VRAM 合计（batch 沿 F32）** | | | **≈ +1684.5 MB** |
| **VRAM 合计（batch 顺带降 F16）** | | | **≈ +656.0 MB** |
| **RAM**（消除 beam=4 的 CPU KV 快照 ≈2184 MB F32 → 近 greedy） | | | **≈ −1640 MB** |

> ⚠️ 上表「batch 沿用 F32」的 +1684 MB 才是与我们当前实现同精度的真实增量；初版 +957 MB **低估**（因初版把当前 cross 误当 f16），且**没有把「当前 cross 是 F32」这个可低成本修复点算进去**——见 §8。
>
> 要点：
> - **cross K/V 是绝对大头，且两点叠加**：① batch 复制 ×4；② 我们当前 cross 用 F32（比半精度多一倍）。若 batch 化**顺手把 cross 降到 F16/bf16**，增量从 +1684 MB 骤降到 **+656 MB**；而**不 batch、仅把当前 F32 cross 降 F16**，单候选 cross 直接 **−245.8 MB**（白捡，见 §8）。
> - **RAM 大幅回落**：当前 beam 把整 KV 快照放 CPU（large 每候选 546 MB F32，beam=4 ≈ 2184 MB），batch 方案移除 → 净降可达 ~1.6 GB。
> - **RAM 明显回落**：当前 beam 把整 KV 快照放 CPU（每候选 146.8 MB F32，beam=4 ≈ 587 MB），batch 方案移除这些 CPU 快照 → 净降 ~440 MB。
> - 权衡：**用小幅度 VRAM 上升（~0.95GB/12GB）换取 RAM 下降 + beam RTF 0.232→~0.03。**

### 6.3 修改边界：是否触及上游源码？

**结论：所有改动落在本仓库 own 代码树（`src/`、`include/`），零改动 vendored 上游 `external/ggml` 内核。**

- 方案触及文件逐一核对归属（`git ls-files`）：
  | 文件 | 归属 |
  |---|---|
  | `include/engine/framework/modules/attention/transformer_blocks.h` | 本仓库 own（`include/`） |
  | `src/framework/modules/attention/transformer_blocks.cpp` | 本仓库 own（`src/`） |
  | `include/engine/framework/runtime/kv_cache.h` | 本仓库 own（`include/`） |
  | `src/framework/runtime/kv_cache.cpp` | 本仓库 own（`src/`） |
  | `src/models/whisper/runtime.cpp` | 本仓库 own（`src/`，当前为 UNTRACKED 新文件，未提交） |
- **`external/ggml` 是 vendored 上游内核**（非子模块、无嵌套 .git，1974 文件全被本仓库 track，含 ggml_cuda 的 fattn 等算子）。本次方案**不需要改动它**——batch 化只需在框架模块层（own 代码）把输入 tensor 扩 B 维、用 ggml **标准算子**（`ggml_repeat` 复制 cross K/V、`mul_mat`、现有 attention/fattn CUDA kernel 均 batch 无关）接线，**不新增/不修改任何 ggml 算子**。
- 参考的 batch 基座（`TransformerBatchedKVCache`/`qwen_causal` batched decode）**同样全是本仓库 own 代码**（`src/framework/runtime/kv_cache.cpp`、`src/framework/modules/transformers/qwen_causal_decode_runtime.cpp`）。
- 唯一需保持克制的上游接触点：若未来要动 `external/ggml` 内的 CUDA 算子（如新增 batch flash-attn kernel）才算"碰上游"，**本方案不需要**。

> ⚠️ 注意：`src/models/whisper/runtime.cpp` 目前是 UNTRACKED（本次会话改造的新文件）。改动它不影响任何已提交/上游内容；若后续需要合入正式分支，请先 `git add` 提交。

---
*（§6 追加于 2026-09-20 22:0x：large-v3 内存/显存量化 + 上游边界判定。）*

---

## 7. faster-whisper（ctranslate2）beam 显存机制：它到底"解决"了 cross 复制吗

> 结论先行：**ctranslate2 并没有"避免"复制 cross K/V——它同样把 encoder memory 的 K/V 复制到 beam 维（`replicate_batches`），与我们 `ggml_repeat` 的方案本质同款。** 它的显存"看着小"不是因为它绕开了复制，而是因为它省的是**别的东西** + 我们此前对比基准错位。本节用源码证据澄清，并据此再校准 §6.2 的估算。

### 7.1 前提纠正：ctranslate2 同样复制 cross K/V 到 beam 维 ✅源码已证

faster-whisper 底层是 CTranslate2（C++）。其 decoder 的 cross-attention 实现 `MultiHeadAttention::forward_merged`（`src/layers/attention.cc`）：

```cpp
// encoder memory 的 K/V 投影只做一次（batch 粒度，不叠 beam）
(*_memory_kv)(*memory, mem_kv);           // 一次 Dense，得到 [batch, ...] 的 memory K/V
...
// 然后显式复制到 beam 维，对齐 batch×beam 的 query
if (beam_size > 1) {
  replicate_batches(memory_keys, beam_size);   // [batch,H,T,d] -> [batch*beam,H,T,d]
  replicate_batches(memory_values, beam_size);
}
static void replicate_batches(StorageView& x, dim_t repeats) {
  x.expand_dims(1);
  ops::Tile(1, repeats)(x);                 // 纯显存内 tile 复制
  x.reshape({x.dim(0)*x.dim(1), x.dim(2), x.dim(3), x.dim(4)});
}
// beam_size 推导
const dim_t beam_size = queries.dim(0) / batch_size_no_beam;
```

**所以"faster-whisper 不复制 cross K/V"是错误前提。** 它做的 `ops::Tile` 与我们要做的 `ggml_repeat` 是**同一件事**：把共享的 encoder memory K/V 复制成 beam 份、对齐到 `batch*beam` 的 decoder query。**它没有回避这个"问题"。**

对照我们的 whisper 实现：`runtime.cpp:619-623` 的 `dec_cross_` 每层 `[1, heads, mem_frames, head_dim]`，mem_frames=`max_source_positions`=1500。batch 化后同样需要 B 份。→ 这一步两边**同量级**，不存在"它更聪明地省了 cross 复制"。

### 7.2 它显存为何不大：真正的机制

| 机制 | 性质 | 对显存的作用 |
|---|---|---|
| **A. encoder memory K/V 投影只算一次**（`_memory_kv` 一次 Dense），复制是 `ops::Tile` 纯拷贝 | ✅源码已证 | 省**算力**（B-1 倍投影），不省显存副本；但避免了我们贪心式"每候选重新投影"的浪费 |
| **B. bf16 精度**（GPU `select_best_compute_type` 优先 bf16>fp16，infer.py:245） | ✅本机工具已证 | 🔶 **修正（v1.1）：在 large + 当前实现下这是量级级差异源，不是"持平"**。faster-whisper decoder 全 bf16；而我们当前 `dec_cross_` 用 **F32**（runtime.cpp:620-621）、仅 self KV 用 F16。large 下 cross 半精度与否差 **245.8 MB/份**（F32 491.5 vs bf16 234.4）。tiny 下这份差仅 ~9 MB（17.6→8.8）→ tiny 持平、large 显著。下文的"它显存低这么多"在 large 场景一多半就来自**它 bf16、我们 cross 用 F32** 这一点。 |
| **C. batch×beam 联合管理，无独立冗余分配**：memory 先 `[batch,..]` 一份，再 `Tile` 到 `[batch*beam,..]`；decoder self-KV 按 `batch*beam` 行独立 | ✅源码已证 | 布局紧凑、无空洞；与我们"beam 作 batch 维"思路一致 |
| **D. 自动降批量保峰值**："Finding optimal batch size"（自测日志已见）——承载不下时自动降 batch/beam，保护峰值不 OOM | ✅自测日志已证 | 显存峰值**受控**，不会硬崩 |
| **E. whisper 句子段短（VAD 切块），decoder 每句实际 token 少** | ⚠️推导 | decoder self-KV 的有效利用低；但 ctranslate2 仍按 max_len 预分配，此项对**峰值分配**帮助有限，主要省**逐句**时间 |

### 7.3 再校准：我们 §6.2 的 +957 MB 是否被高估 / 与它到底差在哪

**关键澄清：§6.2 的 +957 MB 是我们"把 beam 作 batch=B 维、GPU 常驻"**这一方案的**真实增量**（decoder self-KV ×4 +220MB，cross K/V ×4 +737MB）。**

但要注意对比基准错位：
- **我们当前实现的 beam（`transcribe_beam_kv` CPU 往返）**：KV 走 CPU，GPU 上只存 1 份 → **显存反而小**（与 greedy 相同~1348MiB），代价是**慢**。
- **我们规划的 GPU batch beam**：为了快（一次 batch=B 前向），把 KV/cross 全放 GPU ×B 份 → **显存 +957MB**。
- **faster-whisper**：beam 真正在 GPU 内 batch 展开，cross 也复制 beam 份 → **它的 beam 相关显存与本方案同量级**。

所以**实质是**：
1. 我们"显存多很多"的印象，是拿「一次性讨论的 GPU batch beam 方案(+957MB)」和「faster-whisper 的 beam」比，而没和「我们当前 CPU 往返实现(显存小但慢)」比——**三者是三个不同的点**。
2. **faster-whisper 的 beam 显存并没有"更小到哪去"**：同为"cross 复制 beam 份 + decoder KV 按 batch×beam"，它的 cross 那份该多大还是多大（large 下 cross 才是大头）。它真正强的是**性能（GPU 内 batch 前向 + 省算力 + 自动降批）**，不是"显存上绕开了 cross 复制"。

### 7.4 对我们方案的真正启示（比"照抄复制"更有价值）

与其纠结"它怎么省 cross 复制"（它没省），我们真正该学它的是**两件事**：

1. **若显存紧张，别用「长音频整段常驻 batch beam」**——参考 ctranslate2，改成**「短句(VAD切块) + single-encoder + 短 decoder KV」**：encoder 跑一次，cross memory 只一份，decoder 在短句上 batch/beam。句子短 → decoder self-KV 有效长度小 → 总显存/time 都下降。这正是 faster-whisper batch 模式显存观的来源（虽其质量有损，见 §诊断）。
2. **若坚持整段 batch beam（我们的 §6 方案），显存预算按 §6.2 的 +957MB(large, B=4) 计**，并预留"降 batch 保护"（借鉴 ctranslate2 的 auto optimal batch）。

### 7.5 一句话回答

> faster-whisper 没有绕过 cross 复制（`replicate_batches` 同样复制 B 份）；它显存不大的真因是 **encoder memory 只投影一次 + copy 用廉价 tile + bf16 + 自动降批保峰值 + 短句短 KV**，且我们 §6.2 的 +957MB 是"GPU 常驻 batch beam"这一**特定选择**的增量，与它并非"它省、我费"的关系，而是**三种不同实现形态**。要与它公平对比显存，需同基准实测（本节未实测其 beam 峰值，为源码+机制推导；见下）。

---

*（§7 追加于 2026-09-20 22:1x：澄清 faster-whisper cross 复制真相，重校准 §6.2 显存对比基准。）*

---

## 8. 逐块显存对照：它"具体低在哪"（用户问题精答）

> 本轮问题：「既然 batch beam 相同（都要复制 cross B 份），faster-whisper 怎么还能控制那么低？具体是**哪个地方**比我们显存低那么多？」
> 结论先行：**① 真正的差距远没有"那么多"（tiny 实测只差 51 MiB，+4%）；② 它并没有在"复制"上比我们省——beam 的 KV/cross 无论哪家都只占总显存几个百分点，根本不是量级来源；③ 若论 large 级差异，真正的量级差来自「cross 精度 F32 vs bf16」，这是它比我们低的主要实质原因，也是我们自己能低成本修的点。**

### 8.1 先破除直觉：beam 复制从来不是显存大头（tiny 实测）

| 块 | 我们当前(1份) | fast-whisper beam5(GPU内) |
|---|---|---|
| cross K/V | **F32** 17.6 MB | bf16 8.8×5=43.9 MB |
| decoder self KV | F16 2.6 MB | bf16 2.6×5=13.1 MB |
| 合计 KV state | 22.4 MB | ~57 MB |
| **占总实测 VRAM** | 22.4 / 1348 ≈ **1.7%** | 57 / 1297 ≈ **4.4%** |

→ **beam 的 cross/self KV 在总显存里连 5% 都不到**。两者实测总量（1348 vs 1297）的**绝对大头是「模型权重 + encoder/decoder 图工作区 + CUDA 上下文」**，这些两家相当。所以"它复制了 5 份还那么省"的反差，谜底是 **beam 那部分本来就便宜**——它不是"省下了复制"，而是"复制本身就不占地方"。

### 8.2 真正的差距在哪（tiny 实测 51 MiB 差的构成）

实测：我们 1348 MiB vs faster-whisper 1297 MiB = **差 51 MiB（+4%）**。分解（含推导，MiB 级难逐一实测归因）：

| 来源 | 方向 | 量级 |
|---|---|---|
| cross 精度 F32 vs bf16（tiny） | 我们多 | ~9 MB |
| 图 workspace / gallocr 分配（我们整图峰值优先 vs 它对每个算子按需） | 我们多 | 数十 MB |
| CUDA 上下文 / 框架依赖库 | 交叉抵消 | ± |
| beam 复制 | **不是来源** | 两家同量级 |

→ 结论：**tiny 下"它比我们低"主要是 workspace 分配策略 + 小项精度，不是 beam 复制，也不是某个神级优化**。

### 8.3 large-v3 才是"够量级"的差异所在（bf16 vs F32 cross）

large-v3 下，前几轮关注的 +957MB 场景真正拉开差距的是两点，其中**第一点是我们当前就存在的自伤**：

| 项 | 我们当前 | faster-whisper | 差距 |
|---|---|---|---|
| cross K/V 1份 | **F32 491.5 MB** | bf16 234.4 MB | **我们多 +257 MB（×1份）** |
| decoder self KV 1份 | F16 70 MB | bf16 70 MB | 持平（都 2 字节） |
| batch=4 时 cross | 1966 MB（F32） | 937.5 MB（bf16） | 差 1 倍 |

→ **large 下它比我们低的主要实质原因 = cross 用 bf16，我们用 F32。** 我们当前 `dec_cross_`（runtime.cpp:620-621）是 F32，这是历史实现未做半精度优化的遗留，与 beam 机制无关。

### 8.4 对我们最有价值的可落地结论（比"照抄怎么省复制"值钱）

1. **低成本零风险修复：把当前 `dec_cross_` 从 F32 降 F16/bf16**
   - 不 batch、不改 beam 机制，单候选 cross 直接 **−245.8 MB（large）** / −8.8 MB（tiny），且这是纯存储精度、计算仍可在 GPU 内做（ggml 支持 f16 乘算）。
   - **这是"它比我们低"我们唯一能立刻白捡的量级块**。
2. 若还要 batch beam（§6），**cross 顺带 F16** 则增量从 +1684 → +656 MB（见 §6.2 修正）。
3. workspace 分配策略（§8.2）属 ggml 图优化层，收益数十 MB，优先级低。

### 8.5 精度是怎么定的：我们写死 F32 vs 它运行期可配 bf16（用户问题）

**我们侧：编译期硬编码，无外部控制。**
- `dec_cross_` 类型 = `runtime.cpp:620/622` 的 `GGML_TYPE_F32`，**写死在源码里**，与模型权重精度、参数、config 概无关。
- `dec_keys_/dec_values_` = `GGML_TYPE_F16`（615/617）同样硬编码；`kv_memory_` F32（605）。
- cross 计算的投影本身可受 `AttentionConfig::projection_precision` 影响（框架层，理论上可配），但最终经 `ggml_cpy`（648-649）落进 **F32 的 `dec_cross_` buffer** → 目标精度仍被 F32 固定。
- 结论：改精度只能**改 runtime.cpp 源码 + 重编译**，无开关。

**faster-whisper 侧：运行期 `--compute_type` 参数，完全可配。**
- `infer.py:91` `--compute_type` 默认 `"auto"`；578-581：`auto/default` 走 `select_best_compute_type(device)`，否则直接用传入值。
- `select_best_compute_type`（infer.py:202-266）：按 `bfloat16 > float16 > int16 > int8_bfloat16 > int8_float16 > int8_float32 > int8 > float32` 偏好顺序，取设备支持的第一个。RTX 5070 Ti(BF16 支持) → **bf16**。
- 三个官方 GPU bat 均未显式传 `--compute_type` → auto → bf16；低显存 bat 显式 `--compute_type=int8_float16`。
- 结论：它不是"聪明地省显存"，而是**计算类型本来就是开放参数 + 默认偏好 bf16**；用户一行命令行即可改。

> 两相对照：**我们"连可选都没有"（写死 F32），它"开箱即 bf16 且可调"（参数控制）**。这解释了 §8.3 里"它 cross 用 bf16、我们用 F32"的量级差的根因——不是算法差异，是**精度开关的暴露程度**不同。

> 修正记录：§6.2 基数（当前 cross 实为 F32 491.5MB/份，非 f16 245.8）已按实际重算；§7.2-B"bf16 不是差异源"在 large+当前实现下不成立，已改为量级级差异源；本节为其精确到块的量化。

*（§8 追加于 2026-09-20 22:4x：逐块显存对照，修正 §6.2 基数与 §7.2-B bf16 结论。）*

---

## §9 large 实测：cross 链精度可配化落地 + 显存对拍（2026-09-20 23:3x）

### 9.1 代码落地（I1+I2 全字段可配化，非写死）
- `WhisperKVStoragePrecision`（F32/F16/BF16）+ `WhisperRuntimeOptions.cross_kv_precision`（默认 F32 保底）+ `WhisperRuntime` 构造第三参（带默认值）。
- 一个开关统一控**整条 cross 链**：`kv_memory_`（I2，encoder memory）+ `dec_cross_`（I1，投影输出），buffer 在 `build_kv_graphs` 按 `cross_type` 建，填充分支用 `write_tensor_float` 按类型自动转。
- **session/CLI 层可配**：`whisper.cross_kv_precision`（f16/fp16/bf16），CLI 用 `--session-option whisper.cross_kv_precision=f16` 即可切换，无需改代码重编。
- `engine_model_whisper` 编译通过（C++17，勿用指定初始化器 `.field=`）。

### 9.2 large 显存实测（RTX 5070 Ti 12GB，30s 音频 long30s.wav，pynvml 4ms 峰采）
> ⚠️ **必须 `--backend cuda`**：无此参数时 large 走非 GPU 全速路径（RTF 0.88、显存仅 1.36G、权重没全进 GPU）；加 `--backend cuda` 才是 GPU 全速（RTF 0.029、GPU util 88-90%）。tiny 默认就走 GPU，不受影响。

| 配置 | large VRAM 峰值 | 解码 RTF | 依据 |
|---|---|---|---|
| 本框架 **F32** 贪心（large-v3-f16, `--backend cuda`） | **5989 MiB** | 0.029 | ✅ 实测（两次 5989 稳定） |
| 本框架 **F16/BF16** | ❌ **无法运行** | — | 🔴 被 ggml-cuda 否决（见 §9.4） |
| faster-whisper **large-v2 bf16** 贪心（ctranslate2 4.8.2） | **5110 MiB** | 0.73 | ✅ 实测（VAD 分 2 段） |
| faster-whisper **large-v2 bf16** beam=5 | **5105 MiB** | 0.49 | ✅ 实测（beam 对显存峰值几乎无影响） |

> ⚠️ faster-whisper 侧用的是 **large-v2** 的 CT2 模型（v3 CT2 config 格式不完整无法直接跑，见 §9.5）；large-v2/v3 同为 32 层 1280 维，权重规模一致，显存可近似但需标注版本差异。

- **核心对比**：本框架 F32（5989 MiB）比 faster-whisper bf16（~5105 MiB）高 **~880 MiB（+17%）**。§8 的"bf16 更省"结论方向成立，但本框架的 F16 优化因 §9.4 的 ggml 约束**无法落地**，880 MiB 差距只能靠"权重也半精度"或接受现状。
- faster-whisper 解码 RTF 0.49-0.73（慢）源于对 30s 音频的 **VAD 静音分段 + 多语言检测**（切成 2 段各重编码）；本框架贪心一次性转写 RTF 0.029，**快 ~17-25×**。

### 9.3 ✕ 原 9.2 F16 推导值作废
原记"F16 ≈7363 MiB（−238MB）"为**字节级推导**，但沙箱关闭后实测发现 **F16 根本跑不起来**（§9.4），该推导值随可行性一起失效，不再回填。

### 9.4 ★F16/BF16 cross KV 在 CUDA 下被 ggml 硬否决（推翻 I1/I2"可配即用"结论）
- 沙箱关闭后重建 CUDA CLI 成功（含 `cross_kv_precision`），实测 `--session-option whisper.cross_kv_precision=f16` 与 `=bf16` 均报：
  `WhisperCrossKV contains unsupported backend op 'MUL_MAT' at node 1 tensor 'node_1'`
- **根因（代码实证）**：ggml-cuda `ggml_backend_cuda_device_supports_op`（ggml-cuda.cu:5437）的 MUL_MAT 分支 **5530 行**：
  `if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) return false;`
  即 **F16 激活必须配 F16 权重**。我们的 cross 投影 = `kv_memory_(F16/BF16) × cross_qkv(F32, linear_module.cpp:90 make_from_f32)` → 激活 F16、权重 F32 → 命中拒绝 → `validate_backend_graph_supported`（backend.cpp:382）抛异常。
- **为什么 faster-whisper 能 bf16**：它整条链路（encoder memory **和** cross 权重）都是 bf16，不触发此约束；我们只把 memory/激活降半精度、权重仍是 F32，ggml 便拒绝。
- **可执行方向**（需用户决策）：① 接受 F32 现状（880 MiB 差距）；② **把 cross_qkv 权重也按 cross_type 转半精度**（F16 激活配 F16 权重即可通过，能拿到 bf16 级显存，但**改权重精度**、影响质量需权衡）；③ 改 ggml-cuda 支持 F16 激活×F32 权重（cuBLAS 实际可算，但违反"不改官方源码"铁律，排除）。

### 9.5 faster-whisper 环境与 v3 CT2 坑（供复用）
- **Python**：managed py3.14 无 ctranslate2 cp314 wheel，改用 **miniforge3.13** 装 `faster-whisper 1.2.1 + ctranslate2 4.8.2`（cp313 wheel）。
- **ctranslate2 wheel 下载**：19MB 从 aliyun mirror 极慢（120s 仅 2.6MB），需 `curl -C -` **多次断点续传**（19222061B 校验 VALID）。
- **v3 CT2 无法直接跑**：① ct2 目录缺 preprocessor_config.json，需把 hf 目录的（feature_size=128）拷入修 mel 维；② ct2 的 config.json 是原始 transformers 格式（缺 ctranslate2 要求的 `lang_ids/suppress_ids/suppress_ids_begin/alignment_heads`）→ decoder generate 报 `json type_error.302`。→ **改用 v2 ct2（转换格式完整）直接跑通**。
