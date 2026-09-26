#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
words_to_srt.py — 把 audiocpp R2T2 生成的词级时间戳 JSON 转成 SRT 字幕文件。

输入：CLI `--words-out` 产出的 JSON，格式为纯数组：
    [ {"start_sample":6400,"end_sample":15360,"word":"Some","confidence":0}, ... ]

断行策略（R2T2 是纯词级 ASR，无句子结构）：
  1. 词间间隙(gap)超阈值 → 断行（表停顿/句界）
  2. 行内字符数超上限 → 强制断行（中文逐字场景尤为重要）

用法：
  python words_to_srt.py <words.json> <out.srt> [--sample-rate 16000] [--gap-ms 350] [--max-chars 64]
"""
import argparse
import json
import sys


def ms(sample, rate):
    return sample * 1000.0 / rate


def fmt_ts(ms_val):
    ms_val = max(0, ms_val)
    h = int(ms_val // 3600000)
    m = int((ms_val % 3600000) // 60000)
    s = int((ms_val % 60000) // 1000)
    milli = int(ms_val % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{milli:03d}"


def build_lines(words, rate, gap_samples, max_chars):
    lines = []
    cur = []          # 当前行的词列表
    cur_start = None  # 当前行首词 start_sample

    def flush():
        nonlocal cur, cur_start
        if not cur:
            return
        start = cur_start
        end = cur[-1]["end_sample"]
        # 拼接文本：两个相邻词均为非 CJK（字母/数字）时才加空格，中文/日文逐字用无缝
        parts = []
        prev_cjk = True
        for wd in cur:
            w = wd["word"]
            is_cjk = any("\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
                         or "\uac00" <= ch <= "\ud7af" for ch in w)
            if parts and not (prev_cjk or is_cjk):
                parts.append(" ")
            parts.append(w)
            prev_cjk = is_cjk
        text = "".join(parts)
        if not text.strip():
            cur = []
            cur_start = None
            return
        lines.append((start, end, text))
        cur = []
        cur_start = None

    for i, wd in enumerate(words):
        word = wd.get("word", "")
        if not word or not word.strip():
            continue
        start = wd.get("start_sample", 0)
        end = wd.get("end_sample", start)

        # 断行判定
        if cur:
            gap = start - cur[-1]["end_sample"]
            cur_len = sum(len(x["word"]) for x in cur)
            if gap > gap_samples or cur_len + len(word) > max_chars:
                flush()

        if cur_start is None:
            cur_start = start
        cur.append({"word": word, "end_sample": end})

    flush()
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("words_json")
    ap.add_argument("out_srt")
    ap.add_argument("--sample-rate", type=int, default=16000,
                    help="词时间戳的采样率；R2T2/forced-aligner 均基于 16k，默认 16000")
    ap.add_argument("--gap-ms", type=float, default=350.0,
                    help="词间间隙超过该毫秒数则断行（停顿/句界）")
    ap.add_argument("--max-chars", type=int, default=64,
                    help="单行最大字符数（超限强制断行）")
    args = ap.parse_args()

    with open(args.words_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "words" in data:
        # 兼容 server 形态 {words:[...], sample_rate:...}
        if data.get("sample_rate"):
            args.sample_rate = int(data["sample_rate"])
        data = data["words"]

    gap_samples = int(args.gap_ms / 1000.0 * args.sample_rate)
    lines = build_lines(data, args.sample_rate, gap_samples, args.max_chars)

    if not lines:
        print(f"WARN: no words -> empty SRT ({args.words_json})", file=sys.stderr)
        open(args.out_srt, "w", encoding="utf-8-sig", newline="\n").write("")
        return 0

    srt = []
    for idx, (s, e, text) in enumerate(lines, 1):
        srt.append(str(idx))
        srt.append(f"{fmt_ts(ms(s, args.sample_rate))} --> {fmt_ts(ms(e, args.sample_rate))}")
        srt.append(text)
        srt.append("")

    # SRT 写 UTF-8 带 BOM(utf-8-sig)，提升播放器兼容性(尤其中日文字幕)
    with open(args.out_srt, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("\n".join(srt))

    print(f"OK: {len(lines)} lines -> {args.out_srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
