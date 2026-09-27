@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   DanceCut rebuild script (onedir)
echo ============================================
echo.
echo NOTE: onedir is used on purpose. The onefile build has to unpack
echo       ~140MB (incl. 87MB ffmpeg) on every launch, which took ~3 min.
echo       onedir starts in seconds and lets the worker processes share
echo       the bundle instead of unpacking their own copies.
echo.

set PY=C:\Users\wzer\.workbuddy\binaries\python\envs\pyinstaller-build312\Scripts\python.exe
if not exist "%PY%" set PY=python

set SP=C:\Users\wzer\.workbuddy\binaries\python\envs\pyinstaller-build312\Lib\site-packages
if not exist "%SP%\mediapipe\tasks\c\libmediapipe.dll" (
  echo [ERROR] mediapipe is not installed in the build venv:
  echo   %SP%
  echo Run:  "%PY%" -m pip install mediapipe opencv-python numpy pyinstaller
  pause
  exit /b 1
)

if not exist "ffmpeg.exe" (
  echo [ERROR] ffmpeg.exe not found next to this script.
  pause
  exit /b 1
)
if not exist "models\pose_landmarker_full.task" (
  echo [ERROR] models\pose_landmarker_full.task not found.
  pause
  exit /b 1
)

echo [1/2] building ...
"%PY%" -m PyInstaller ^
  --onedir --windowed --name DanceCut --noconfirm --clean ^
  --distpath "%~dp0.build\dist" ^
  --workpath "%~dp0.build\work" ^
  --specpath "%~dp0.build" ^
  --add-binary "%~dp0ffmpeg.exe;." ^
  --add-data "%~dp0models\pose_landmarker_full.task;." ^
  --add-data "%~dp0models\hand_landmarker.task;." ^
  --add-data "%SP%\mediapipe\tasks\c\libmediapipe.dll;mediapipe\tasks\c" ^
  --hidden-import mediapipe.tasks.c ^
  --hidden-import mediapipe.tasks.python.vision ^
  --hidden-import mediapipe.tasks.python.vision.pose_landmarker ^
  --hidden-import mediapipe.tasks.python.vision.hand_landmarker ^
  --exclude-module matplotlib --exclude-module pandas --exclude-module scipy ^
  --exclude-module PIL --exclude-module playwright --exclude-module bilibili_api ^
  --exclude-module setuptools --exclude-module pip --exclude-module unittest ^
  --exclude-module pydoc --exclude-module doctest --exclude-module pytest ^
  --exclude-module IPython --exclude-module sqlite3 --exclude-module distutils ^
  "%~dp0dance_cut.py"
if errorlevel 1 (
  echo [ERROR] build failed
  pause
  exit /b 1
)

echo [2/2] installing output ...
if exist "%~dp0DanceCut" rmdir /s /q "%~dp0DanceCut"
move /y "%~dp0.build\dist\DanceCut" "%~dp0DanceCut" >nul
echo.
echo Done:  %~dp0DanceCut\DanceCut.exe
echo Keep the whole DanceCut folder together when copying.
echo Self-test:  DanceCut\DanceCut.exe --self-test
pause
