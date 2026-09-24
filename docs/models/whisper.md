---
# Whisper (multilingual ASR)

audio.cpp runs OpenAI Whisper (whisper-tiny architecture) as a native offline ASR
model, initially on CPU. The runtime accepts a local Hugging Face safetensors
checkpoint layout: `config.json` + `multilingual.tiktoken` + `model.safetensors`.

| Field | Value |
|---|---|
| Family | `whisper` |
| Task | `asr` |
| Mode | `offline`, `streaming` |
| Languages | `en`, `zh`, `ja`, `ko`, `es`, `fr`, `de`, `ru` (via `<|lang|>` prompt; default `en`) |
| Audio | WAV; converted to mono 16 kHz internally |
| Output | Transcript text; with `whisper.timestamps=true`, `speech_segments` + `word_timestamps` (segment- and word-level time spans) |
| Streaming | Sliding-window (audio chunks → incremental `partial_text`); decoder itself also streams per-token via KV-cache |
| Timestamps | Exposed via `whisper.timestamps=true` (segments + words) |

## Status

`wip`. The full transcription pipeline is verified end-to-end against the
reference implementation: encoder (WhisperEmbeddingModule, 6 heads from
`config.json`) and a KV-cache streaming decoder greedy tour reproduce the
reference output token-for-token. The decoder builds a single per-token graph
(encoder KV budget + cached-tail blocks) reused across steps, so it no longer
re-runs the whole prefix sequence every step.

## Install

There is no hosted download yet. Two local layouts are supported:

**GGUF (recommended):** a single self-contained `whisper-tiny-*.gguf` produced by
`audiocpp_gguf` embeds the package spec plus `config.json` and
`multilingual.tiktoken`, so it needs no side files and no `model_specs` directory.

```text
whisper-tiny-f16.gguf
```

**Safetensors:** the three official `openai/whisper-tiny` files in one directory:

```text
<model-dir>/
  config.json
  multilingual.tiktoken
  model.safetensors
```

Point the CLI at the GGUF file or at the directory with `--model`.

## Build A GGUF

Convert the safetensors layout to a portable F16 or Q8_0 GGUF (both verified to
transcribe the validation clip):

```bash
audiocpp_gguf \
  --input /path/to/whisper-tiny/model.safetensors \
  --root /path/to/whisper-tiny \
  --output /path/to/whisper-tiny-f16.gguf \
  --type f16 \
  --family whisper \
  --overwrite
```

`--root` supplies `config.json` and `multilingual.tiktoken`, which are embedded
as sidecars. Use `--type q8_0` for a smaller quantized container. Inspect the
result with `audiocpp_gguf --inspect /path/to/whisper-tiny-f16.gguf`.

## Backends

CPU (default) and **CUDA** are both supported. CUDA requires an `EnableCuda=ON` build; point `--backend cuda` to select it (the ggml graphs, KV-cache streaming, beam, timestamps, translate, long-audio chunking all run unmodified — verified 2026-09-20 on RTX 5070 Ti laptop, sm_120a, CUDA 12.8 + ggml-cuda).

Long-audio throughput (327.6 s input, same whisper-tiny model, same GPU):

| Implementation | Alg | Wall | RTF |
|---|---|---|---|
| this framework, CUDA | greedy | 1.21 s | 0.0018 |
| this framework, CPU | greedy | 4.84 s | 0.0135 |
| faster-whisper-tiny (CT2) | greedy | 4.03 s | 0.0123 |
| faster-whisper-tiny (CT2) | beam=5 | 10.3 s | 0.0314 |
| this framework, CUDA | beam=4 | 76.7 s | 0.232 |
| this framework, CUDA | beam=5 | 84.3 s | 0.257 |

Memory & VRAM peaks (327.6 s input, whisper-tiny, RTX 5070 Ti / 12 GB):

| Implementation | Alg | RAM peak | VRAM peak |
|---|---|---|---|
| this framework, CPU | greedy | 143 MiB | — |
| this framework, CUDA | greedy | 532 MiB | 1348 MiB |
| this framework, CUDA | beam=4 | 593 MiB | 1348 MiB |
| faster-whisper-tiny (CT2) | beam=5 | 507 MiB | 1297 MiB |

Note: greedy (the default) is ~6.8× faster than faster-whisper at identical model/size/GPU. Beam is slower than faster-whisper's GPU-internal batching because this framework round-trips each beam candidate's KV state to CPU (see the CUDA plan, §3); greedy avoids that cost entirely. Since the round-tripped KV lives on the CPU, beam does **not** raise VRAM (greedy and beam both peak at 1348 MiB) — a `GPU-internal constant-batch beam` (CUDA plan §7.6) would move that KV to VRAM, cutting RAM and bringing beam RTF from 0.232 into faster-whisper's ~0.03 range.

## CLI

```bash
audiocpp_cli \
  --task asr \
  --family whisper \
  --model /path/to/whisper-tiny \
  --audio /path/to/speech.wav \
  --backend cpu \
  --option whisper.language=en
```

Supported request options:

| Option | Type | Default | Description |
|---|---|---|---|
| `whisper.language` | string | `en` | Whisper ISO-639 language code used for the `<|lang|>` prompt token. Use `auto` for single-step language detection.
| `whisper.max_tokens` | int | `0` (→ 64) | Maximum generated tokens per utterance; capped at `max_target_positions` (448). |
| `whisper.task` | enum | `transcribe` | `transcribe` (ASR) or `translate` (transcribe into the target language; Open AI semantics output English). |
| `whisper.timestamps` | bool | `false` | Emit word and speech segments with timestamps (`speech_segments` + `word_timestamps` populated, `word_timestamps` derived by proportional token interpolation). When enabled, audio up to 30 s produces multi-segment output; longer audio is split on quiet-energy boundaries (default) or Silero VAD (`audio_chunk_mode=vad`). |
| `whisper.beam_size` | int | `0` | Beam search width. `0`/`1` => greedy KV-cache streaming; `>=2` => beam search over per-beam KV state (single-batch forward per candidate, no framework batch rewrite). Higher width can improve transcription at higher latency. |
| `whisper.audio_chunk_mode` | enum | `quiet_energy` | Long-audio segmentation strategy: `quiet_energy` (pure energy-valley, zero VAD dependency) or `vad` (real Silero VAD activity detection, loaded lazily from `vad_model_path`). Each chunk is capped at 30 s. |
| `whisper.vad_model_path` | string | `assets/framework/models/silero_vad` | Directory containing `silero_vad_16k.safetensors`, used only when `audio_chunk_mode=vad`. |

## Streaming

Whisper exposes a sliding-window streaming session (mode `streaming`). Audio
chunks are accumulated, resampled to 16 kHz mono, and transcribed in fixed
windows of `audio_chunk_seconds` (default **5 s** so partials arrive promptly;
Whisper's encoder is capped at 30 s per window). Each completed window produces
an incremental `partial_text` delta; `finalize()`/`finish_stream()` flushes any
trailing partial window and returns the full transcript.

```bash
audiocpp_cli --task asr --mode streaming --family whisper \
  --model /path/to/whisper-tiny-f16.gguf \
  --audio /path/to/speech.wav --backend cuda \
  --request-option audio_chunk_seconds=5
```

The streaming session reuses the same offline `transcribe` path per window, so
all request options (`whisper.language`, `whisper.task`, `whisper.max_tokens`,
`whisper.beam_size`, etc.) behave identically to the offline CLI. Server
streaming is available via the `/v1/audio/transcriptions/live` endpoint
(chunked body) returning `partial_text` deltas.

## Frontend

The log-mel frontend is configured for whisper alignment: HTK mel scale +
periodic Hann (`Kokoro` family) window + whisper log10 floor normalization,
padded/truncated to 30 s at 16 kHz. This was validated frame-by-frame against
the reference with RMSE below `1e-3`.

## Validation

The `librispeech_test_clean_6930-75918-0000.wav` fragment transcribes to the
expected `[Music]` tokens, matching the independent numpy reference. Additional
language samples can be added over time.

## Known limitations

- Decoder uses KV-cache streaming (single per-token forward graph, incremental
  cross/self K/V caches). Each token is decoded with incremental attention caches
  rather than a full-prefix rebuild; results match the numpy reference
  token-for-token.
- Timestamp/segment output is available (`whisper.timestamps=true`); within a
  30 s window it maps `<|t|>` tokens to `speech_segments` and `word_timestamps`.
  Word-level stamps are derived by splitting each segment's tokens on the token
  byte stream (a token that starts with a space begins a new word) and
  interpolating the segment's `[start, end]` proportionally to per-word token
  count (openai-whisper's attention-free fallback). For English this yields
  per-word stamps that tile the segment end-to-end. Space-less scripts (CJK)
  produce one word per segment.
- Long audio (>30 s) is split automatically on quiet-energy boundaries, then
  reassembled into global-timestamp `speech_segments` and concatenated text.
  Timestamp accuracy across a split boundary is best-effort on tiny.
- `whisper.task=translate` follows Open AI semantics: it always produces the
  target **English** transcription of the source language — there is no
  "translate to Chinese" mode. To go Japanese → Chinese you must run a separate
  translation model (e.g. the Sakura JP→CN model used elsewhere), not whisper.
  Verified on a synthesized Japanese clip (`temp/whisper_probe/ja_sample.wav`):
  `language=ja` transcribes to Japanese and `language=ja,task=translate` switches
  output to English, but whisper-tiny's non-English quality is poor (it decodes
  to repetitive/garbled text), so cross-lingual translation on tiny is not
  production-useful.
- `whisper.language=auto` runs online single-step language detection over the
  declared 8 languages before decoding.
- Language tokens strongly bias the output language. On the tiny model, English
  input forced to non-English languages (e.g. `zh`/`ja`/`ko`/`ru`) yields
  lower-quality, language-flavoured output by design; English remains the
  primary, best-validated direction. All 8 declared languages map to the
  official token ids (verified against openai-whisper's tokenizer).
- Beam search (`whisper.beam_size>=2`) walks multiple candidates with
  independent per-beam KV state on the same single-batch graph (no framework
  batch rewrite), scoring with length-normalized cumulative log-prob times a
  sqrt-unique-ratio repetition discount. Quality tracks the underlying model:
  on whisper-tiny, non-timestamp (greedy-notimestamps) decoding of noisy or
  music-only clips can fall into repetitive loops — prefer
  `whisper.timestamps=true` (beam + timestamps yields stable, closed segments,
  e.g. "I'll see you in the next video. Bye.") or stick with the default greedy
  for such clips. A larger model removes most of this instability.
