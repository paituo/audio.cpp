@echo off
setlocal enabledelayedexpansion
rem ============================================================
rem R2T2 batch subtitle generator: drag one or more media files
rem onto this script to produce a .srt next to each media file.
rem pipeline: media -> ffmpeg 16k mono wav (staging dir) -> ONE
rem R2T2 ASR + forced aligner process (model loaded ONCE for all
rem files) -> words_to_srt.py per file -> .srt in each media dir.
rem Compared to the old per-file loop this saves one full model
rem load (~9-10s) per extra file, so multi-file drags are far
rem faster. Requirements: ffmpeg and python on PATH; CUDA build
rem with the Qwen3-ForcedAligner model. Edit config block below.
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
set "WORK=%ROOT%\temp\r2t2_subtitle_batch"
set "PYTHON=python"
set "BACKEND=cuda"
rem optional language override applied to ALL files, e.g. Chinese/English
set "LANG="
set "SR=16000"

rem ---- sanity checks ----
if not exist "%CLI%"    ( echo ERROR: CLI not found: %CLI%    & exit /b 1 )
if not exist "%MODEL%"  ( echo ERROR: R2T2 model not found: %MODEL% ^& exit /b 1 )
if not exist "%ALIGNER%" ( echo ERROR: aligner not found: %ALIGNER% ^& exit /b 1 )

if "%~1"=="" ( echo Drag one or more media files onto this .bat & exit /b 1 )
if not exist "%WORK%"       mkdir "%WORK%"
if not exist "%WORK%\in"   mkdir "%WORK%\in"
if not exist "%WORK%\out"  mkdir "%WORK%\out"
del /q /f "%WORK%\in\*.wav"        >nul 2>&1
del /q /f "%WORK%\out\*.json"      >nul 2>&1
del /q /f "%WORK%\batch*.json"      >nul 2>&1
del /q /f "%WORK%\map.txt"          >nul 2>&1

set /a N=0

rem ---- phase 1: stage each media to a unique 16k mono wav ----
:staging
if "%~1"=="" goto batch
set "SRC=%~1"
if not exist "%SRC%" ( echo SKIP file not found: %SRC% & shift & goto staging )
for %%F in ("%SRC%") do set "NAMESTEM=%%~nF"
set /a N+=1
set "STEM=!NAMESTEM!_!N!"
set "OUTDIR=%~dp1"
set "WAV=%WORK%\in\!STEM!.wav"
set "SRT=%OUTDIR%!NAMESTEM!.srt"

echo.
echo === %SRC%
echo [1/2] extract audio to 16k mono wav ...
ffmpeg -y -i "%SRC%" -ac 1 -ar %SR% -c:a pcm_s16le "!WAV!" >nul 2>&1
if errorlevel 1 ( echo   ffmpeg failed, skip & shift & goto staging )
rem remember wav-stem -> srt destination so the srt lands next to media
>>"%WORK%\map.txt" echo !STEM!^|!SRT!
shift
goto staging

rem ---- phase 2: ONE R2T2 process for all staged wavs ----
:batch
echo.
echo [2/2] R2T2 ASR + forced aligner over %N% file(s), model loaded once ...
if "%LANG%"=="" (
  "%CLI%" --task asr --family confucius4_r2t2 --model "%MODEL%" --backend %BACKEND% --batch-audio-dir "%WORK%\in" --session-option "confucius4_r2t2.forced_aligner_model_path=%ALIGNER%" --words-out "%WORK%\batch.json"
) else (
  "%CLI%" --task asr --family confucius4_r2t2 --model "%MODEL%" --backend %BACKEND% --batch-audio-dir "%WORK%\in" --language %LANG% --session-option "confucius4_r2t2.forced_aligner_model_path=%ALIGNER%" --words-out "%WORK%\batch.json"
)
if errorlevel 1 ( echo   R2T2 batch failed & goto cleanup )

rem ---- phase 3: convert each word file to its .srt ----
set /a DONE=0
for /f "usebackq tokens=1,* delims=^|" %%A in ("%WORK%\map.txt") do (
  set "JS=%%~A"
  set "S=%%~B"
  set "WORDS=%WORK%\batch_!JS!.json"
  "%PYTHON%" "%CONV%" "!WORDS!" "!S!" --sample-rate %SR%
  if errorlevel 1 ( echo   SRT step failed for !JS! ) else ( echo   =^> !S! & set /a DONE+=1 )
)

rem ---- cleanup staging files ----
:cleanup
del /q /f "%WORK%\in\*.wav"   >nul 2>&1
del /q /f "%WORK%\batch*.json" >nul 2>&1
rmdir /q /s "%WORK%\in"   >nul 2>&1
rmdir /q /s "%WORK%\out"  >nul 2>&1
echo.
echo Done. !DONE!/%N% subtitle(s) written. R2T2 loaded the model once.
endlocal