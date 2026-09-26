@echo off
setlocal enabledelayedexpansion
rem ============================================================
rem R2T2 subtitle generator: drag one or more media files onto
rem this script to produce a .srt subtitle next to each media.
rem pipeline: media -> ffmpeg 16k mono wav -> R2T2 ASR + forced
rem aligner (word timestamps) -> words_to_srt.py -> .srt
rem Requirements: ffmpeg and python on PATH; CUDA build with the
rem Qwen3-ForcedAligner model. Edit config block below as needed.
rem ============================================================
chcp 65001 >nul
rem --- run under UTF-8 code page so CJK words are written correctly ---
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"

rem ---- config (edit these as needed) ----
set "CLI=%ROOT%\build\windows-cuda-release\bin\audiocpp_cli.exe"
set "MODEL=%ROOT%\models\Confucius4-R2T2"
set "ALIGNER=%ROOT%\models\Qwen3-ForcedAligner-0.6B-GGUF"
set "CONV=%SCRIPT_DIR%words_to_srt.py"
set "WORK=%ROOT%\temp\r2t2_subtitle"
set "PYTHON=python"
set "BACKEND=cuda"
rem optional language override, e.g. set "LANG=Chinese" (empty=auto)
set "LANG="
set "SR=16000"

rem ---- sanity checks ----
if not exist "%CLI%"   ( echo ERROR: CLI not found: %CLI%   & exit /b 1 )
if not exist "%MODEL%"  ( echo ERROR: R2T2 model not found: %MODEL% ^& exit /b 1 )
if not exist "%ALIGNER%" ( echo ERROR: aligner not found: %ALIGNER% ^& exit /b 1 )

if "%~1"=="" ( echo Drag one or more media files onto this .bat & exit /b 1 )
if not exist "%WORK%" mkdir "%WORK%"

set /a COUNT=0

:loop
if "%~1"=="" goto done
set "SRC=%~1"
if not exist "%SRC%" ( echo SKIP file not found: %SRC% & shift & goto loop )
for %%F in ("%SRC%") do set "NAME=%%~nF"
set "OUTDIR=%~dp1"
set "WAV=%WORK%\%NAME%_%RANDOM%.wav"
set "WORDS=%WORK%\%NAME%_%RANDOM%.words.json"
set "SRT=%OUTDIR%%NAME%.srt"

echo.
echo === %SRC%
echo [1/3] extract audio to 16k mono wav ...
ffmpeg -y -i "%SRC%" -ac 1 -ar %SR% -c:a pcm_s16le "%WAV%" >nul 2>&1
if errorlevel 1 ( echo   ffmpeg failed, skip & goto cleanup )

echo [2/3] R2T2 ASR + forced aligner ...
if "%LANG%"=="" (
  "%CLI%" --task asr --family confucius4_r2t2 --model "%MODEL%" --backend %BACKEND% --audio "%WAV%" --session-option "confucius4_r2t2.forced_aligner_model_path=%ALIGNER%" --request-option return_timestamps=true --words-out "%WORDS%"
) else (
  "%CLI%" --task asr --family confucius4_r2t2 --model "%MODEL%" --backend %BACKEND% --audio "%WAV%" --language %LANG% --session-option "confucius4_r2t2.forced_aligner_model_path=%ALIGNER%" --request-option return_timestamps=true --words-out "%WORDS%"
)
if errorlevel 1 ( echo   ASR failed, skip & goto cleanup )

echo [3/3] write SRT ...
"%PYTHON%" "%CONV%" "%WORDS%" "%SRT%" --sample-rate %SR%
if errorlevel 1 ( echo   SRT step failed & goto cleanup )
echo   =^> %SRT%

:cleanup
if exist "%WAV%" del /q /f "%WAV%" >nul 2>&1
if exist "%WORDS%" del /q /f "%WORDS%" >nul 2>&1
:next
set /a COUNT+=1
shift
goto loop

:done
echo.
echo Done. %COUNT% file(s) processed.
endlocal
