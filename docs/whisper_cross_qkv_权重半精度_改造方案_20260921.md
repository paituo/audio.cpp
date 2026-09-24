# Whisper cross_qkv 权重半精度化 —— 改造方案（2026-09-21）

> 关联：`docs/whisper_beam_optimization.md` §9.4 方向②「把 cross_qkv 权重也按 cross_type 转半精度」的落地施工方案。
> 目标：让本框架拿到 faster-whisper 同等的 bf16 级显存（实测 large-v2：7655 → ~3780 MiB，减半 50.6%），
> 同时不违反「不改官方 ggml 源码」铁律。

---

## 0. 结论先行

**只需改一处**：`src/models/whisper/runtime.cpp` 的 `load_block` 中 cross-attention 的 **packed K/V 权重（`qkv_weight`/`qkv_bias`）存储类型**，从写死的 `TensorStorageType::F32` 改为**跟随 `cross_kv_precision_`**。

- `cross_kv_precision=F32` → 权重 F32（**现状，行为零变化**）
- `cross_kv_precision=F16` → 权重 **F16**（满足 ggml-cuda.cu:5530「F16 激活须配 F16 权重」）
- `cross_kv_precision=BF16` → 权重 **BF16**（对齐 faster-whisper bf16 全链路；需本机 GPU 支持 bf16）

**为什么只改这一组权重**：整个 decoder 里，只有这条 packed K/V 投影的**输入是 cross memory（F16/BF16 激活）**。query(×q_weight)、output(×out_weight)、self-attn、FFN 的输入都是 decoder 内部 F32 激活，权重保持 F32 没有任何类型冲突，改它们毫无必要（也不该动，避免无谓精度损失）。

---

## 1. 根因（已在 §9.4 实证）

- cross K/V 投影：`kv_memory_(F16/BF16) × cross_qkv 权重(F32)` 的 `MUL_MAT`。
- ggml-cuda.cu:5530：`if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) return false;`（`a`=权重=cross_qkv，`b`=激活=memory）。
- 激活 F16、权重 F32 → 命中拒绝 → `validate_backend_graph_supported` 抛 `WhisperCrossKV contains unsupported backend op 'MUL_MAT'`。
- faster-whisper/CTranslate2 没这问题：它的 compute_type 把**权重与激活（含 cross KV 缓存）一视同仁统一转半精度**，二者同型，从根上不撞约束。

**本方案 = 照 faster-whisper 的「权重与激活同型」做法，但只作用于真正需要的那一条 cross 投影，改动最小、精度损失最可控。**

---

## 2. 代码定位（现状）

| 位置 | 现状 | 作用 |
|---|---|---|
| `runtime.cpp:319-320` | `qkv_weight = store.make_from_f32(..., F32, pack_kv(ck,cv))` | cross K/V 权重，**× memory** |
| `runtime.cpp:321-322` | `qkv_bias   = store.make_from_f32(..., F32, pack_kv(cbk,cbv))` | cross K/V bias |
| `runtime.cpp:315-318` | `q_weight/q_bias = make_from_f32(F32)` | cross **query** 权重，× decoder query（F32）→ **不改** |
| `runtime.cpp:323-325` | `out_weight = load_tensor(Native)` | cross output，× F32 context → **不改** |
| `runtime.cpp:633` | `kv_memory_ = make_tensor(cross_type, {1,mem_frames,D})` | cross memory 激活，已随 `cross_kv_precision_` |
| `runtime.cpp:648-651` | `dec_cross_[i] = make_tensor(cross_type, ...)` | cross K/V 投影输出缓存，已随 `cross_kv_precision_` |
| `linear_module.cpp:90` | `ggml_mul_mat(weights.weight.tensor, matrix_input.tensor)` | 就是上面那条冲突的 MUL_MAT |

> `cross_kv_precision_` 已由构造期选项（`session.cpp:106-120` 的 `whisper.cross_kv_precision` → `WhisperRuntimeOptions.cross_kv_precision`）解析，`Impl` 构造 `: cross_kv_precision_(options.cross_kv_precision)` 初始化（runtime.cpp:397），`load_decoder_weights` 在 423 行调用——**时机上完全可用**。

---

## 3. 改动清单（最小集）

### 3.1 二选一：把 cross 权重存储类型传递进 load 链

`load_block`/`load_decoder_weights` 目前是自由函数，拿不到 `cross_kv_precision_`。任选其一：

- **3.1A（推荐）**：给 `load_decoder_weights`/`load_block` 增加参数 `assets::TensorStorageType cross_kv_storage = assets::TensorStorageType::F32`，在 `Impl` 构造里由 `cross_kv_precision_` 映射后传入：
  ```cpp
  assets::TensorStorageType cross_kv_storage = assets::TensorStorageType::F32;
  switch (cross_kv_precision_) {
      case WhisperKVStoragePrecision::F16:  cross_kv_storage = assets::TensorStorageType::F16;  break;
      case WhisperKVStoragePrecision::BF16: cross_kv_storage = assets::TensorStorageType::BF16; break;
      case WhisperKVStoragePrecision::F32:
      default:                              cross_kv_storage = assets::TensorStorageType::F32;  break;
  }
  decoder_weights_ = load_decoder_weights(*decoder_store_, source, cfg, cross_kv_storage);
  ```
- **3.1B**：把 storage 作为 `DecoderWeights` 返回结构之外的一个成员传入 `load_block`（较绕，不推荐）。

### 3.2 核心改动：`load_block` 里 cross K/V 权重改用传入的 storage

```cpp
// 原：
w.cross_attention.qkv_weight = store.make_from_f32(
    TensorShape::from_dims({2 * D, D}), assets::TensorStorageType::F32, pack_kv(ck, cv));
w.cross_attention.qkv_bias = store.make_from_f32(
    TensorShape::from_dims({2 * D}), assets::TensorStorageType::F32, pack_kv(cbk, cbv));

// 改（storage 参数传入，默认 F32 → 行为不变）：
w.cross_attention.qkv_weight = store.make_from_f32(
    TensorShape::from_dims({2 * D, D}), cross_kv_storage, pack_kv(ck, cv));
w.cross_attention.qkv_bias = store.make_from_f32(
    TensorShape::from_dims({2 * D}), cross_kv_storage, pack_kv(cbk, cbv));
```

> `make_from_f32`（backend_weight_store.h:143）经 `values_to_bytes`（:351）已原生支持 `TensorStorageType::F16/BF16` 的 `ggml_fp32_to_fp16_row` / `ggml_fp32_to_bf16_row` 转换——**无需动 BackendWeightStore**，只换个枚举值。

### 3.3 无其它改动（关键确认）

- **输出 cast 已桥接**：`build_key_value` 的投影输出是 F32（`linear_module.cpp:100` wrap），写回 `dec_cross_[i]`（F16/BF16）时 `runtime.cpp:676` 用的 `ggml_cpy(ctx.ggml, kv.key.tensor, dec_cross_[i].key.tensor)` 自动 cast F32→cross_type，无需额外处理。
- **query/output/self-attn/FFN 权重一律不动**：它们乘的激活都是 F32，无类型冲突。
- **不改任何 ggml 官方源码**（符合铁律）。
- `--session-option whisper.cross_kv_precision=...` 的解析、`WhisperKVStoragePrecision` 枚举、`WhisperRuntimeOptions` 字段**全部复用**，无需扩展。

---

## 4. 显存收益预估（large-v2，实测锚点）

| 档位 | cross 链精度（memory+权重） | 加载显存 | 相对 F32 |
|---|---|---|---|
| F32（现状） | F32 × F32 | 7655 MiB（实测） | 基准 |
| F16 | F16 × F16 | ~3730 MiB（实测锚点） | **−50.6%** |
| BF16 | BF16 × BF16 | ~3780 MiB（实测锚点） | **−50.6%** |

> 注：上面是 faster-whisper/CT2 同体量的整模型加载值。本框架只改 cross 一条投影 + 它下游的 cross KV 缓存（本就随 minor），收益为 **cross 权重大小**（large-v3：2D×D 每层 ×层数，D=1280，cross qkv 2×1280×1280×4B→2B ≈ −4.2 MB/层，全 32 层 ≈ −134 MB 常驻）+ 已有 memory 半精度收益。**cross 权重减半是这次新增的增量收益**；整模型级的大头（所有线性层）faster-whisper 才全转，本方案为「针对冲突链最小改动」不追求全模型减半。

---

## 5. 风险与权衡

| 风险 | 说明 | 缓解 |
|---|---|---|
| **精度损失** | cross K/V 权重转 F16/BF16，影响 attention 投影精度，可能影响翻译质量（尤其 F16 尾数只有 10 bit） | 提供档位开关，质量敏感场景回 F32；实测对拍（见 §6） |
| **BF16 后端依赖** | BF16 memory 已在 §9.4 实测被拒，需确认是「权重 F32 导致」还是「BF16 本身不支持」；`use_batched_cublas_bf16 = BF16 && bf16_mma_hardware_available(cc)` | 方案落地时**先单测 BF16 权重 × BF16 memory**；若本机 GPU 无 bf16 MMA，则 BF16 档回退 F16 或保持 F32，并在日志明示 |
| **make_from_f32 字节对齐** | F16/BF16 逐元素转换，无 block 对齐问题（非量化），安全 | 已知安全（backend_weight_store.h:360-372） |
| **cuBLAS 组合** | 权重 F16/BF16 × 激活同型+cublas：2803-2805 显示 f16/bf16 权重走 batched cuBLAS，需实测确认本机走通 | §6 验证 |

---

## 6. 验证计划（严格对照「实测 > 推导」铁律）

1. **单元验证（最省，复用现有 tiny/large 探针）**：小改后跑 `--session-option whisper.cross_kv_precision=f16`，确认不再报 `unsupported backend op 'MUL_MAT'`，转录输出与 F32 档对比 token 一致性。
2. **BF16 档单测**：确认 bf16 权重×bf16 memory 在本机 GPU 是否走通（`bf16_mma_hardware_available`），否则按 §5 回退策略处理。
3. **显存归档**：pynvml 4ms 峰采，对比 F32/F16/BF16 三档加载显存，回填本文档 §4。
4. **质量对拍**：同一段音频，F32 vs F16（vs 若能跑则 BF16）转写文本 diff，量化质量影响；与 faster-whisper bf16 结果对比（见 `docs/whisper_beam_optimization.md` §9）。
5. **回归**：默认档（F32）行为零变化断言 —— token 级一致。

---

## 7. 落地优先级建议

1. **先做 3.1A + 3.2 最小改动**（F16 档先落地，风险最低、cu:5530 明确支持）。
2. **再单测 BF16 档**硬件支持（30 分钟内可判）。
3. 若质量/显存对拍符合预期，再考虑是否把「权重跟随」作为长期默认或暴露独立开关 `whisper.cross_qkv_storage`（可选，非必须）。

> 本方案不改任何官方 ggml 源码、不动 encoder/self-attn/FFN，改动局限于 whisper runtime 的 cross 权重加载一处，默认行为（F32）严格不变，是可安全落地的增量优化。
