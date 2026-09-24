# download_confucius4_r2t2.ps1
# ============================================================================
# One-time / resume downloader for Confucius4-R2T2 (audio.cpp dedicated GGUF)
#
# WHY THIS SCRIPT EXISTS:
#   - The official HuggingFace hosts are NOT reachable from this machine
#     without a proxy; and when a proxy is used it throttles concurrency.
#   - Modelscope-hosted `netease-youdao/Confucius4-R2T2-GGUF` is built for
#     vLLM / llama.cpp (only ~310 tensors) and is INCOMPATIBLE with audio.cpp:
#       audio.cpp reports `missing tensor: thinker.audio_tower.conv2d1.weight`.
#   - The compatible package is the audio.cpp-native single-file GGUF
#     `davidxifeng/Confucius4-R2T2-gguf` (707 tensors, embeds an audio.cpp
#     model spec), reachable via hf-mirror.com (direct) OR huggingface.co (proxy).
#
# This script downloads that GGUF and then fills in the tokenizer/config files
# from the Modelscope official safetensors repo. Idempotent: skips work done.
#
# Usage:
#   # Direct (no proxy, hf-mirror + parallel segments - fastest on this box)
#   powershell -ExecutionPolicy Bypass -File scripts/download_confucius4_r2t2.ps1
#
#   # Force proxy (e.g. v2rayN etc.) - auto-reads system proxy when -AutoProxy
#   powershell -ExecutionPolicy Bypass -File scripts/download_confucius4_r2t2.ps1 -Proxy http://127.0.0.1:10808
#   powershell -ExecutionPolicy Bypass -File scripts/download_confucius4_r2t2.ps1 -AutoProxy
#
# Downstream behavior:
#   - Direct  mode -> GGUF fetched from hf-mirror with N parallel Range segments.
#   - Proxy   mode -> GGUF fetched from huggingface.co over the proxy with a
#                     SINGLE resumable curl stream (measured: this proxy
#                     rejects concurrent connections, so parallel segments die).
#   - Proxy   mode with -KeepMirror -> stays on hf-mirror (still single stream).
#
# NOTE (learned pitfalls, do not regress):
#   - Do NOT try to use processor_config.json from the Modelscope repo: that
#     file does not exist there and returns {"Code":10990101007,...}; having it
#     in the model directory makes audio.cpp fail with
#     `missing required json key: feature_size`. Correct frontend params live in
#     preprocessor_config.json (feature_size=128, hop_length=160, n_fft=400).
#   - The model directory must contain ONLY ONE .gguf (the loader picks the
#     first .gguf in lexical order).
# ============================================================================

[CmdletBinding()]
param(
    # Override target directory (default: <repo>/models/Confucius4-R2T2)
    [string]$Target = "",
    # Number of parallel Range segments (direct mode only; default 6)
    [int]$Segments = 6,
    # Explicit proxy URL, e.g. "http://127.0.0.1:10808". Empty => no proxy
    # (unless -AutoProxy is set).
    [string]$Proxy = "",
    # Read the proxy from the Windows system proxy settings automatically.
    [switch]$AutoProxy,
    # When set, proxy mode still fetches the GGUF from hf-mirror instead of
    # huggingface.co (default: proxy mode uses huggingface.co).
    [switch]$KeepMirror
)

$ErrorActionPreference = "Stop"

# --- resolve paths -----------------------------------------------------------
$RepoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Target)) {
    $Target = Join-Path $RepoRoot "models\Confucius4-R2T2"
}
$Target = [System.IO.Path]::GetFullPath($Target)

# --- proxy resolution ----------------------------------------------------------
# Priority: explicit -Proxy  >  -AutoProxy (system settings)  >  no proxy
$UseProxy = $false
$ProxyUrl = ""
if (-not [string]::IsNullOrWhiteSpace($Proxy)) {
    $UseProxy = $true
    $ProxyUrl = $Proxy
} elseif ($AutoProxy) {
    try {
        $is = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -ErrorAction Stop
        if ($is.ProxyEnable -eq 1 -and -not [string]::IsNullOrWhiteSpace($is.ProxyServer)) {
            $UseProxy = $true
            if ($is.ProxyServer -match "^https?://") { $ProxyUrl = $is.ProxyServer }
            else { $ProxyUrl = "http://" + $is.ProxyServer }
        }
    } catch {
        Write-Warning "Failed to read system proxy: $($_.Exception.Message)"
    }
}

# --- endpoints ----------------------------------------------------------------
$DirectGgufUrl = "https://hf-mirror.com/davidxifeng/Confucius4-R2T2-gguf/resolve/main/r2t2-q8_0.gguf"
$ProxyGgufUrl  = "https://huggingface.co/davidxifeng/Confucius4-R2T2-gguf/resolve/main/r2t2-q8_0.gguf"
$GgufSize      = 2477512064L
$GgufName      = "r2t2-q8_0.gguf"
$CfgsBase      = "https://modelscope.cn/models/netease-youdao/Confucius4-R2T2/resolve/master"

# Config files sourced from the Modelscope official safetensors repo.
# NOTE: processor_config.json is deliberately OMITTED (see header).
$ConfigFiles = @(
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json"
)

function Write-Step([string]$msg) {
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# --- mode banner ----------------------------------------------------------------
if ($UseProxy) {
    $ggufSrc = if ($KeepMirror) { "hf-mirror.com" } else { "huggingface.co" }
    Write-Step ("Proxy MODE  : using proxy {0} ; GGUF from {1} (single resumed stream; this proxy rejects parallel connections)" -f $ProxyUrl, $ggufSrc)
} else {
    Write-Step ("DIRECT MODE : no proxy ; GGUF from hf-mirror.com ($Segments parallel Range segments)")
}

# --- 1. create target ----------------------------------------------------------
New-Item -ItemType Directory -Force -Path $Target | Out-Null
Write-Step "Target: $Target"

# --- curl invocation helper -----------------------------------------------------
function Invoke-Segment([string]$url, [string]$range, [string]$out) {
    # returns exit code
    $arg = @("-sL", "--retry", "4", "--retry-delay", "2", "--max-time", "5400")
    if (-not [string]::IsNullOrWhiteSpace($range)) {
        $arg += @("-H", ("Range: bytes={0}" -f $range))
    } else {
        # resume from current file offset via -C -
        $arg += @("-C", "-")
    }
    if ($UseProxy) { $arg += @("-x", $ProxyUrl) }
    $arg += @("-o", $out, $url)
    & "curl.exe" @arg 2>$null
    return $LASTEXITCODE
}

# --- 2. GGUF download -------------------------------------------------------------
$ggufPath = Join-Path $Target $GgufName
$fullyDone = $false
if (Test-Path $ggufPath) {
    $have = (Get-Item $ggufPath).Length
    if ($have -eq $GgufSize) {
        $fullyDone = $true
        Write-Step "GGUF already complete ($have bytes). Skipping download."
    } else {
        Write-Step "GGUF partial ($have / $GgufSize bytes). Resuming."
    }
}

if (-not $fullyDone) {
    $ggufUrl = if ($UseProxy) { if ($KeepMirror) { $DirectGgufUrl } else { $ProxyGgufUrl } } else { $DirectGgufUrl }

    if ($UseProxy) {
        # ---- single resumed stream (proxy rejects concurrency) ----
        Write-Step "Downloading $GgufName ($GgufSize bytes) in single resumed stream over proxy..."
        $rc = Invoke-Segment $ggufUrl "" $ggufPath
        if ($rc -ne 0) { Write-Warning "curl exit code: $rc (incomplete is resumable - re-run to continue)" }
    } else {
        # ---- parallel Range segments ----
        Write-Step "Downloading $GgufName ($GgufSize bytes) via hf-mirror with $Segments parallel segments..."
        $start = 0L
        if (Test-Path $ggufPath) { $start = (Get-Item $ggufPath).Length }
        $remaining = $GgufSize - $start
        $segSize = [long][Math]::Ceiling($remaining / $Segments)
        if ($segSize -lt 1) { $segSize = 1 }

        $jobs = New-Object System.Collections.ArrayList
        $segFiles = New-Object System.Collections.ArrayList
        for ($i = 0; $i -lt $Segments; $i++) {
            $s = $start + $i * $segSize
            if ($s -ge $GgufSize) { break }
            $e = $s + $segSize - 1
            if ($e -ge $GgufSize) { $e = $GgufSize - 1 }
            $part = Join-Path $Target ("seg_{0}.part" -f $i)
            [void]$segFiles.Add($part)
            $p = Start-Process -FilePath "curl.exe" `
                -ArgumentList @("-sL","--retry","4","--retry-delay","2","--max-time","5400",
                                "-H",("Range: bytes={0}-{1}" -f $s, $e),
                                "-o", $part, $ggufUrl) `
                -PassThru -WindowStyle Hidden
            [void]$jobs.Add($p)
        }
        foreach ($j in $jobs) { $j.WaitForExit() | Out-Null }
        $jobs | ForEach-Object { if ($_.ExitCode -ne 0) { Write-Warning "curl segment exit code: $($_.ExitCode)" } }

        if ($segFiles.Count -gt 0) {
            $fs = [System.IO.File]::Open($ggufPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try {
                foreach ($seg in $segFiles) {
                    if (Test-Path $seg) {
                        $bytes = [System.IO.File]::ReadAllBytes($seg)
                        $fs.Write($bytes, 0, $bytes.Length)
                    }
                }
            } finally {
                $fs.Close()
                $fs.Dispose()
            }
            foreach ($seg in $segFiles) { if (Test-Path $seg) { Remove-Item $seg -Force -ErrorAction SilentlyContinue } }
        }
    }

    $final = (Get-Item $ggufPath).Length
    Write-Step "GGUF now $final / $GgufSize bytes"
    if ($final -eq $GgufSize) {
        Write-Step "GGUF download complete."
    } else {
        Write-Warning "GGUF incomplete ($final/$GgufSize). Re-run this script to resume."
    }

    # verify GGUF magic
    if ($final -ge 4) {
        $fs = [System.IO.File]::OpenRead($ggufPath)
        $buf = New-Object byte[] 4
        [void]$fs.Read($buf, 0, 4)
        $magic = [System.Text.Encoding]::ASCII.GetString($buf)
        $fs.Close()
        if ($magic -eq "GGUF") {
            Write-Step "GGUF magic OK: 'GGUF'"
        } else {
            Write-Warning "GGUF magic invalid: '$magic' (file corrupt? delete and re-download)"
        }
    }
}

# --- 3. config / tokenizer files from Modelscope ---------------------------------
# NOTE: these config files come from Modelscope, which is a CHINA-mainland CDN.
#   - Direct  mode: fetch them directly (fast).
#   - Proxy   mode: we STILL fetch them directly (NOT through the proxy). Measured:
#     the local proxy exits via an overseas IP, and Modelscope CDN returns
#     HTTP 403 on merges.txt / tokenizer.json / vocab.json for foreign IPs.
#     So do NOT route config downloads through the proxy.
Write-Step "Checking config + tokenizer files from Modelscope official repo (direct, no proxy)..."
foreach ($cf in $ConfigFiles) {
    $dst = Join-Path $Target $cf
    if (Test-Path $dst) {
        $sz = (Get-Item $dst).Length
        if ($sz -gt 0) { Write-Host "  keep  $cf ($sz bytes)"; continue }
    }
    Write-Host "  fetch $cf ..."
    $url = "$CfgsBase/$cf"
    try {
        $wc = New-Object System.Net.WebClient
        # force direct connection even when proxy mode is active (Modelscope 403s foreign IPs)
        $wc.Proxy = $null
        $wc.DownloadFile($url, $dst)
        Write-Host "       -> $((Get-Item $dst).Length) bytes"
    } catch {
        Write-Warning "Failed to fetch $cf : $($_.Exception.Message)"
    }
}

# --- 4. final report -------------------------------------------------------------
Write-Step "Done. Directory contents:"
Get-ChildItem $Target | Sort-Object Name | ForEach-Object {
    $mb = if ($_.PSIsContainer) { "<dir>" } else { [math]::Round($_.Length / 1MB, 1) }
    Write-Host ("  {0,-28} {1,10} MB" -f $_.Name, $mb)
}

$gg = Join-Path $Target $GgufName
$gS = if (Test-Path $gg) { (Get-Item $gg).Length } else { 0 }
if ($gS -eq $GgufSize) {
    Write-Host ""
    Write-Host "All set. Run offline ASR with:" -ForegroundColor Green
    Write-Host "  audiocpp_cli --task asr --family confucius4_r2t2 --model `"$Target`" --backend cuda --audio <voice_16k.wav> --text-out out.txt" -ForegroundColor Green
    Write-Host "Streaming (LSP, chunk 320ms):" -ForegroundColor Green
    Write-Host "  audiocpp_cli --task asr --mode streaming --family confucius4_r2t2 --model `"$Target`" --backend cuda --audio <voice_16k.wav> --session-option confucius4_r2t2.chunk_size_ms=320 --text-out out.txt" -ForegroundColor Green
} else {
    Write-Warning "GGUF not complete yet ($gS / $GgufSize). Re-run to resume."
}
